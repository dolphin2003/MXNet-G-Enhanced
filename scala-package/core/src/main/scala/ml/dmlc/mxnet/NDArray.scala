package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.ref.WeakReference

/**
 * NDArray API of mxnet
 * @author Yizhi Liu, Yuan Tang
 */
object NDArray {
  private val logger = LoggerFactory.getLogger(classOf[NDArray])

  private[mxnet] val DTYPE_NATIVE_TO_MX: Map[Class[_ >: Float with Int with Double], Int] = Map(
    classOf[Float] -> 0,
    classOf[Double] -> 1,
    classOf[Int] -> 4
  )

  private[mxnet] val DTYPE_MX_TO_NATIVE: Map[Int, Class[_ >: Float with Int with Double]] = Map(
    0 -> classOf[Float],
    1 -> classOf[Double],
    2 -> classOf[Float],
    3 -> classOf[Int],
    4 -> classOf[Int]
  )

  private val functions: Map[String, NDArrayFunction] = initNDArrayModule()

  private def addDependency(froms: Array[NDArray], tos: Array[NDArray]): Unit = {
    froms.foreach { from =>
      val weakRef = new WeakReference(from)
      tos.foreach { to =>
        to.dependencies.put(from.handle, weakRef)
        // we add all dep's dep to prevent (recursively) recomputing at runtime.
        to.dependencies ++= from.dependencies
      }
    }
  }

  // Definition of internal functions.
  // Internal binary function
  def invokeBinaryFunc(funcName: String,
                       lhs: NDArray, rhs: NDArray,
                       out: NDArray = null): NDArray = {
    var output = out
    val function = functions(funcName)
    require(function != null, s"invalid function name $funcName")
    require(output == null || output.writable, "out must be writable")
    function match {
      case BinaryNDArrayFunction(handle: FunctionHandle, acceptEmptyMutate: Boolean) =>
        if (output == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          output = new NDArray(newEmptyHandle())
          addDependency(Array(lhs, rhs), Array(output))
        }
        checkCall(_LIB.mxFuncInvoke(handle,
          Array(lhs.handle, rhs.handle),
          Array[MXFloat](),
          Array(output.handle)))
      case _ => throw new IllegalArgumentException(s"call $funcName as binary function")
    }
    output
  }

  def invokeUnaryFunc(funcName: String, src: NDArray, out: NDArray = null): NDArray = {
    var output = out
    val function = functions(funcName)
    require(function != null, s"invalid function name $funcName")
    require(output == null || output.writable, "out must be writable")
    function match {
      case UnaryNDArrayFunction(handle: NDArrayHandle, acceptEmptyMutate: Boolean) =>
        if 