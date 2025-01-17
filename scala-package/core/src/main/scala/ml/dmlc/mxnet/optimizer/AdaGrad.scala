
package ml.dmlc.mxnet.optimizer

import ml.dmlc.mxnet.NDArrayConversions._
import ml.dmlc.mxnet.{NDArray, Optimizer}

/**
 * AdaGrad optimizer as described in Matthew D. Zeiler, 2012.
 * http://arxiv.org/pdf/1212.5701v1.pdf
 *
 * @author Yuan Tang, Yizhi Liu
 *
 * @param learningRate Step size.
 * @param epsilon A small float number to make the updating processing stable.
 *                Default value is set to 1e-7.
 * @param rescaleGradient rescaling factor of gradient.
 * @param wd L2 regularization coefficient add to all the weights
 */
class AdaGrad(val learningRate: Float = 0.05f, val rescaleGradient: Float = 1.0f,
              val epsilon: Float = 1e-7f, val wd: Float = 0.0f) extends Optimizer {

  /**
   * Update the parameters.
   * @param index An unique integer key used to index the parameters
   * @param weight weight ndarray
   * @param grad grad ndarray
   * @param state NDArray or other objects returned by initState
   *              The auxiliary state used in optimization.
   */
  override def update(index: Int, weight: NDArray, grad: NDArray, state: AnyRef): Unit = {
    val lr = this.learningRate

    val resdGrad = rescaleGradient * grad
    val history = state.asInstanceOf[NDArray]

    val gradSquared = resdGrad * resdGrad
    history += gradSquared
    gradSquared.dispose()

    val newWeight = (-lr * (resdGrad / NDArray.sqrt(history + this.epsilon) + this.wd * weight))
      .disposeDepsExcept(resdGrad, history, weight)
    weight += newWeight
    newWeight.dispose()

    resdGrad.dispose()
  }

  override def createState(index: Int, weight: NDArray): NDArray = {
    NDArray.zeros(weight.shape, weight.context)
  }

  // Dispose the state it created
  override def disposeState(state: AnyRef): Unit = {
    if (state != null) {
      state.asInstanceOf[NDArray].dispose()
    }
  }
}