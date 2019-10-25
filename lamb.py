from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.keras import optimizers


class Lamb(optimizers.Adam):
    """LAMB Optimizer for Tensorflow 2.0

    It is an implement of LAMB Optimizer
    (Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes) (https://arxiv.org/abs/1904.00962)

    This implement is based on `tensorflow.keras.optimizers.Adam`.
    We override following methods as the document showing
    (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#write_a_customized_optimizer)

        `_resource_apply_dense`
        `_resource_apply_sparse`
        `get_config`

        `_create_slots` is same as `Adam._create_slots`
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        lambda_=0.01,
        amsgrad=False,
        name='Lamb',
        excluded_names=['bias', 'layernorm'],
        **kwargs
    ):
        super(Lamb, self).__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=False,
            name=name,
            **kwargs,
        )
        self._excluded_names = excluded_names
        self._set_hyper('lambda_', lambda_)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Lamb, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lambda_ = array_ops.identity(self._get_hyper('lambda_', var_dtype))
        lr = (
            apply_state[(var_device, var_dtype)]['lr_t'] *
            (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))
        )
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
                lambda_=lambda_,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (
            (apply_state or {}).get((var_device, var_dtype))
            or self._fallback_apply_state(var_device, var_dtype)
        )

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        beta_1_power = coefficients['beta_1_power']
        beta_2_power = coefficients['beta_2_power']
        lr_t = coefficients['lr_t']
        beta_1_t = coefficients['beta_1_t']
        beta_2_t = coefficients['beta_2_t']
        epsilon = coefficients['epsilon']
        lambda_ = coefficients['lambda_']

        # $m_{t} = \beta_{1} m_{t-1} + (1-\beta_{1}) g_{t}$
        m_t = beta_1_t * m + (1 - beta_1_t) * grad

        # $v_{t} = \beta_{2} v_{t-1} + (1-\beta_{2}) g_{t}^{2}$
        v_t = beta_2_t * v + (1 - beta_2_t) * math_ops.pow(grad, 2)

        # $m_{t} = m_{t} / (1-\beta_{1}^{t})$
        # $v_{t} = v_{t} / (1-\beta_{2}^{t})$
        # compute ratio $r_{t} = \frac{m_{t}}{\sqrt{v_{t}}+\epsilon}$
        # r_t = (
        #     math_ops.sqrt(1 - beta_2_power) * m_t / (1 - beta_1_power) /
        #     (math_ops.sqrt(v_t) + epsilon)
        # )
        r_t = ((m_t / (1 - beta_1_power)) / (math_ops.sqrt(v_t / (1 - beta_2_power)) + epsilon))

        # Add L2 regularization
        var_name = var.name
        # No add L2 to LayerNorm and bias
        if not self._is_excluded_variable(var_name):
            r_t += lambda_ * var

        w_norm = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(var)))
        g_norm = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(r_t)))

        # https://github.com/ymcui/LAMB_Optimizer_TF/blob/a804c2f2995cda9a4f6b804ab445e19fc4a1036f/optimization.py#L259
        # Note: Here are two choices for scaling function \phi(z)
        # minmax:   \phi(z) = min(max(z, \gamma_l), \gamma_u)
        # identity: \phi(z) = z
        # The authors does not mention what is \gamma_l and \gamma_u
        # UPDATE: after asking authors, they provide me the code below.
        # ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
        #      math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)
        ratio = array_ops.where(
            math_ops.greater(w_norm, 0),
            array_ops.where(math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0),
            1.0,
        )

        # $x_{t+1}^{(i)} =
        #   x_{t}^{(i)} - \eta_{t} \frac{\phi(|x_{t}^{(i)}|)}{|r_{t}^{(i)}+\lambda x_{t}^{(i)}|}
        #   (r_{t}^{(i)} + \lambda x_{t}^{(i)})$
        var_update = var - lr_t * ratio * r_t

        return control_flow_ops.group(*[
            var.assign(var_update),
            m.assign(m_t),
            v.assign(v_t),
        ])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (
            (apply_state or {}).get((var_device, var_dtype))
            or self._fallback_apply_state(var_device, var_dtype)
        )

        beta_1_power = coefficients['beta_1_power']
        beta_2_power = coefficients['beta_2_power']
        lr_t = coefficients['lr_t']
        beta_1_t = coefficients['beta_1_t']
        beta_2_t = coefficients['beta_2_t']
        one_minus_beta_1_t = coefficients['one_minus_beta_1_t']
        one_minus_beta_2_t = coefficients['one_minus_beta_2_t']
        epsilon = coefficients['epsilon']
        lambda_ = coefficients['lambda_']

        # $m_{t} = \beta_{1} m_{t-1} + (1-\beta_{1}) g_{t}$
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * one_minus_beta_1_t
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # $v_{t} = \beta_{2} v_{t-1} + (1-\beta_{2}) g_{t}^{2}$
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * one_minus_beta_2_t
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        # $m_{t} = m_{t} / (1-\beta_{1}^{t})$
        # $v_{t} = v_{t} / (1-\beta_{2}^{t})$
        # compute ratio $r_{t} = \frac{m_{t}}{\sqrt{v_{t}}+\epsilon}$
        # r_t = (
        #     math_ops.sqrt(1 - beta_2_power) * m_t / (1 - beta_1_power) /
        #     (math_ops.sqrt(v_t) + epsilon)
        # )
        r_t = ((m_t / (1 - beta_1_power)) / (math_ops.sqrt(v_t / (1 - beta_2_power)) + epsilon))

        # Add L2 regularization
        var_name = var.name
        # No add L2 to LayerNorm and bias
        if not self._is_excluded_variable(var_name):
            r_t = state_ops.assign_add(
                r_t,
                lambda_ * var,
                use_locking=self._use_locking,
            )
            with ops.control_dependencies([r_t]):
                pass

        w_norm = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(var)))
        g_norm = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(r_t)))

        # https://github.com/ymcui/LAMB_Optimizer_TF/blob/a804c2f2995cda9a4f6b804ab445e19fc4a1036f/optimization.py#L259
        #
        # Note: Here are two choices for scaling function \phi(z)
        # minmax:   \phi(z) = min(max(z, \gamma_l), \gamma_u)
        # identity: \phi(z) = z
        # The authors does not mention what is \gamma_l and \gamma_u
        # UPDATE: after asking authors, they provide me the code below.
        # ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
        #      math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)
        ratio = array_ops.where(
            math_ops.greater(w_norm, 0),
            array_ops.where(math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0),
            1.0,
        )
        with ops.control_dependencies([ratio]):
            pass

        # $x_{t+1}^{(i)} =
        #   x_{t}^{(i)} - \eta_{t} \frac{\phi(|x_{t}^{(i)}|)}{|r_{t}^{(i)}+\lambda x_{t}^{(i)}|}
        #   (r_{t}^{(i)} + \lambda x_{t}^{(i)})$
        var_update = state_ops.assign_sub(
            var,
            lr_t * ratio * r_t,
            use_locking=self._use_locking,
        )
        with ops.control_dependencies([var]):
            pass

        return control_flow_ops.group(*[
            var_update,
            m_t,
            v_t,
        ])

    def _is_excluded_variable(self, name):
        for excluded in self._excluded_names:
            if excluded in name.lower():
                return True
        return False

    def get_config(self):
        config = super(Lamb, self).get_config()
        config.update(
            {
                'learning_rate': self._serialize_hyperparameter('learning_rate'),
                'decay': self._serialize_hyperparameter('decay'),
                'beta_1': self._serialize_hyperparameter('beta_1'),
                'beta_2': self._serialize_hyperparameter('beta_2'),
                'lambda_': self._serialize_hyperparameter('lambda_'),
                'epsilon': self.epsilon,
                'amsgrad': self.amsgrad,
            }
        )
        return config
