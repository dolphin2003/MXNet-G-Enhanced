package ml.dmlc.mxnet.examples.neuralstyle.end2end

import ml.dmlc.mxnet.Executor
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Shape

/**
 * @author Depeng Liang
 */
object ModelVgg19 {
  case class ConvExecutor(executor: Executor, data: NDArray, dataGrad: NDArray,
                      style: Array[NDArray], content: NDArray, argDict: Map[String, NDArray])

  def getVggSymbol(prefix: String, contentOnly: Boolean = false): (Symbol, Symbol) = {
    // declare symbol
    val data = Symbol.Variable(s"${prefix}_data")
    val conv1_1 = Symbol.Convolution(s"${prefix}_conv1_1")()(Map("data" -> data,
                            "num_filter" -> 64, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu1_1 = Symbol.Activation(s"${prefix}_relu1_1")()(Map("data" -> conv1_1,
                            "act_type" -> "relu"))
    val conv1_2 = Symbol.Convolution(s"${prefix}_conv1_2")()(Map("data" -> relu1_1,
                            "num_filter" -> 64, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu1_2 = Symbol.Activation(s"${prefix}_relu1_2")()(Map("data" -> conv1_2,
                            "act_type" -> "relu"))
    val pool1 = Symbol.Pooling(s"${prefix}_pool1")()(Map("data" -> relu1_2 , "pad" -> "(0,0)",
                            "kernel" -> "(2,2)", "stride" -> "(2,2)", "pool_type" -> "avg"))
    val conv2_1 = Symbol.Convolution(s"${prefix}_conv2_1")()(Map("data" -> pool1,
                            "num_filter" -> 128, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu2_1 = Symbol.Activation(s"${prefix}_relu2_1")()(Map("data" -> conv2_1,
                            "act_type" -> "relu"))
    val conv2_2 = Symbol.Convolution(s"${prefix}_conv2_2")()(Map("data" -> relu2_1,
                            "num_filter" -> 128, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu2_2 = Symbol.Activation(s"${prefix}_relu2_2")()(Map("data" -> conv2_2,
                            "act_type" -> "relu"))
    val pool2 = Symbol.Pooling("pool2")()(Map("data" -> relu2_2 , "pad" -> "(0,0)",
                            "