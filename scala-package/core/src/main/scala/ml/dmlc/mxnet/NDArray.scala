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
        if (output == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          output = new NDArray(newEmptyHandle())
          addDependency(Array(src), Array(output))
        }
        checkCall(_LIB.mxFuncInvoke(handle,
          Array(src.handle),
          Array[MXFloat](),
          Array(output.handle)))
      case _ => throw new IllegalArgumentException(s"call $funcName as unary function")
    }
    output
  }

  /**
   * Invoke this function by passing in parameters
   *
   * @param args Positional arguments of input scalars and NDArray
   * @param kwargs: Key-value arguments for functions. e.g.,
   *            out: NDArray or tuple of NDArray, optional
   *            Output NDArray, used to hold the output result.
   * @return The result NDArray(tuple) of result of computation.
   */
  def invokeGenericFunc(funcName: String,
                        args: Array[Any] = null,
                        kwargs: Map[String, Any] = null): Array[NDArray] = {
    var mutateVars: Array[NDArray] = null
    val realKwargs =
      if (kwargs != null && kwargs.contains("out")) {
        val out = kwargs("out")
        mutateVars =
          if (out.isInstanceOf[NDArray]) {
            Array(kwargs("out").asInstanceOf[NDArray])
          } else {
            kwargs("out").asInstanceOf[Array[NDArray]]
          }
        kwargs - "out"
      } else {
        kwargs
      }
    val function = functions(funcName)
    require(function != null, s"invalid function name $funcName")
    function match {
      case GenericNDArrayFunction(handle: FunctionHandle,
                                  acceptEmptyMutate: Boolean,
                                  nMutateVars: Int,
                                  useVarsRange: Range,
                                  scalarRange: Range) =>
        require(mutateVars == null || nMutateVars == mutateVars.length,
          s"expect $nMutateVars in $funcName")
        val useVars = useVarsRange.map(args(_).asInstanceOf[NDArray]).toArray
        val scalarVars = scalarRange.map(args(_).asInstanceOf[MXFloat]).toArray
        if (mutateVars == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          mutateVars = Array.fill[NDArray](nMutateVars)(new NDArray(newEmptyHandle()))
          addDependency(useVars, mutateVars)
        }
        val (numKwargs: Int,
              kwargKeys: Option[Array[Array[Byte]]],
              kwargVals: Option[Array[Array[Byte]]]) =
          if (realKwargs == null) {
            (0, None, None)
          } else {
            (realKwargs.size,
              Some(realKwargs.keys.map(_.getBytes("ASCII") ++ Array(0.toByte)).toArray),
              Some(realKwargs.values.map(_.toString.getBytes("ASCII") ++ Array(0.toByte)).toArray))
          }
        checkCall(_LIB.mxFuncInvokeEx(handle,
          useVars.map(_.handle),
          scalarVars,
          mutateVars.map(_.handle).array,
          numKwargs, kwargKeys.orNull, kwargVals.orNull))
      case _ => throw new IllegalArgumentException(s"call $funcName as generic function")
    }
    mutateVars
  }

  /**
   * Return a new empty handle.
   * Empty handle can be used to hold result
   *
   * @return a new empty ndarray handle
   */
  private def newEmptyHandle(): NDArrayHandle = {
    val hdl = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayCreateNone(hdl))
    hdl.value
  }

  /**
   * Return a new handle with specified shape and context.
   * Empty handle is only used to hold results
   *
   * @return a new empty ndarray handle
   */
  private def newAllocHandle(shape: Shape,
                             ctx: Context,
                             delayAlloc: Boolean): NDArrayHandle = {
    val hdl = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayCreate(
      shape.toArray,
      shape.length,
      ctx.deviceTypeid,
      ctx.deviceId,
      if (delayAlloc) 1 else 0,
      hdl))
    hdl.value
  }

  /**
   * Wait all async operation to finish in MXNet
   * This function is used for benchmark only
   */
  def waitall(): Unit = {
    checkCall(_LIB.mxNDArrayWaitAll())
  }

  // Create a NDArray function from the FunctionHandle.
  private def makeNdarrayFunction(handle: FunctionHandle): (String, NDArrayFunction) = {
    val NDARRAY_ARG_BEFORE_SCALAR = 1
    val ACCEPT_EMPTY_MUTATE_TARGET = 1 << 2
    // Get the property of NDArray
    val nUsedVars = new MXUintRef
    val nScalars = new MXUintRef
    val nMutateVars = new MXUintRef
    val typeMask = new RefInt
    checkCall(_LIB.mxFuncDescribe(handle, nUsedVars, nScalars, nMutateVars, typeMask))
    val acceptEmptyMutate = (typeMask.value & ACCEPT_EMPTY_MUTATE_TARGET) != 0
    // infer type of the function
    val ndarrayArgBeforeScalar = (typeMask.value & NDARRAY_ARG_BEFORE_SCALAR) != 0
    val useVarsRange: Range =
      if (ndarrayArgBeforeScalar) 0 until nUsedVars.value
      else nScalars.value until (nUsedVars.value + nScalars.value)
    val scalarRange: Range =
      if (ndarrayArgBeforeScalar) nUsedVars.value until (nUsedVars.value + nScalars.value)
      else 0 until nScalars.value
    // Get the information from the function
    val name = new RefString
    val desc = new RefString
    val numArgs = new MXUintRef
    val argNames = ListBuffer[String]()
    val argTypes = ListBuffer[String]()
    val argDescs = ListBuffer[String]()

    checkCall(_LIB.mxFuncGetInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs))
    val paramStr = Base.ctypes2docstring(argNames, argTypes, argDescs)
    val docStr = s"${name.value}\n${desc.value}\n\n$paramStr\n"
    logger.debug("NDArray function defination:\n{}", docStr)
    if (nMutateVars.value == 1 && nUsedVars.value == 2 && nScalars.value == 0) {
      (name.value, BinaryNDArrayFunction(handle, acceptEmptyMutate))
    } else if (nMutateVars.value == 1 && nUsedVars.value == 1 && nScalars.value == 0) {
      (name.value, UnaryNDArrayFunction(handle, acceptEmptyMutate))
    } else {
      (name.value, GenericNDArrayFunction(handle,
                                          acceptEmptyMutate,
                                          nMutateVars.value,
                                          useVarsRange,
                                          scalarRange))
    }
  }

  // List and add all the ndarray functions to current module.
  private def initNDArrayModule(): Map[String, NDArrayFunction] = {
    val functions = ListBuffer[FunctionHandle]()
    checkCall(_LIB.mxListFunctions(functions))
    functions.map(makeNdarrayFunction).toMap
  }

  /**
   * One hot encoding indices into matrix out.
   * @param indices An NDArray containing indices of the categorical features.
   * @param out The result holder of the encoding.
 