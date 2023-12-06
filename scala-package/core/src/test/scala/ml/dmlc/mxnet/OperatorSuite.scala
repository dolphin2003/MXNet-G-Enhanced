package ml.dmlc.mxnet

import ml.dmlc.mxnet.CheckUtils._

import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{Matchers, BeforeAndAfterAll, FunSuite}
import org.scalacheck.Gen

import scala.collection.mutable

class OperatorSuite extends FunSuite with BeforeAndAfterAll
  with Matchers with GeneratorDrivenPropertyChecks {
  private def checkElementwiseSumWithShape(shape: Shape, n: Int) = {
    // forward
    val inputs = (0 until n).map(i => Symbol.Variable(s"arg $i"))
    val out = Symbol.ElementWiseSum(name = "esum")(inputs: _*)()
    val arr = (0 until n).map(_ => Random.uniform(-10, 10, shape))
    val arrGrad = (0 until n).map(_ => NDArray.empty(shape))
    val exec = out.bind(Context.cpu(), args = arr, argsGrad = arrGrad)
    exec.forward()
    val forwardOutput = exec.outputs(0)
    val forwardOutputExpected = arr.reduce(_ + _)
    assert(reldiff(forwardOutput, forwardOutputExpected) < 1e-6)

    // backward
    val outGrad = Random.uniform(-10, 10, shape)
    exec.backward(outGrad)
    arrGrad.foreach(grad => assert(grad === outGrad))
  }

  test("elementwise sum") {
    checkElementwiseSumWithShape(Shape(5, 5, 3), 4)
    forAll (Gen.choose(1, 4), Gen.choose(1, 8)) { (dim, n) =>
      forAll (Gen.listOfN(dim, Gen.choose(1, Math.pow(1000, 1.0 / dim).toInt))) { shape =>
        checkElementwiseSumWithShape(Shape(shape), n)
      }
    }
  }

  // TODO: checkSliceChannel

  private def checkConcatWithShape(shapes: Seq[Shape], dimension: Int, skipSecond: Boolean) = {
    // if skipSecond is true, second argument will not have gradient.
    // it is to test #1130
    // forward
    val targetDim = shapes.map(_(dimension)).sum

    val inputs = (0 until shapes.size).map(i => Symbol.Variable(s"arg$i"))
    val out = Symbol.Concat(name = "conc")(inputs: _*)(Map("dim" -> dimension))
    val arr = shapes.map { shape =>
      val nd = NDArray.empty(shape)
      nd.set(shape(dimension))
    }
    val arrNp = arr.map(_.copy())
    val arrGrad = shapes.map(NDArray.empty(_))
    val argNames = out.listArguments()
    val dictGrad =
      (argNames zip arrGrad).filter { case (name, d) =>
        !skipSecond || name != "arg1"
      }.toMap

    val args = out.listArguments()
    val (argShapes, outShapes, auxShapes) = out.inferShape(args.zip(shapes).toMap)
    val outGrad = NDArray.empty(outShapes(0))
    val exec1 = out.bind(Context.cpu(), arr, dictGrad)
    exec1.forward()
    val out1 = exec1.outputs(0)
    // FIXME: only support concatenate at axis0
    val ret = NDArray.concatenate(arr)
    assert(out1 === ret)

    // backward
    out1.copyTo(outGrad)
    outGrad += 1
    exec1.backward(outGrad)
    argNames.zipWithIndex.foreach { case (name, i) =>
      if (!skipSecond || name != "arg1") {
        val grad = dictGrad(name)
        val npGrad = arrNp(i)
        assert(grad === npGrad + 1)
      }
    }
  }

  test("concat") {
    val merge = Array(2, 3, 4, 5, 6)
    forAll (Gen.choose(2, 5)) { dim =>
      val shapes = mutable.ArrayBuffer.empty[Shape]
      for (i <- 0 until dim) {
        shapes += Shape(merge(i), 2)
      }
      // TODO: check dimension > 0
      checkConcatWithShape(shapes, 0, skipSecond = true)
      checkConcatWithShape(shapes, 0, skipSecond = false)
    }
  }

  private def checkRegression(model: Symbol,
                              forward: Float => Float,
                              backward: (Float, Float) => Float) = {
    val shape = Shape(3, 1)
    val arrData = Random.uniform(-1, 1, shape)
    val arrLabel = Random.uniform(0, 1, Shape(shape.head))
    val arrGrad = NDArray.empty(shape)
    val exec1 = model.bind(Context.cpu(),
      args = Array(arrData, arrLabel), argsGrad = Map("data" -> arrGrad))
    exec1.forward()
    assert(exec1.outputs(0).shape === shape)
    val out1 = exec1.outputs(0).toArray
    val npout = arrData.toArray.map(forward(_))
    assert(CheckUtils.reldiff(npout, out1) < 1e-6f)

    exec1.backward()
    // arrData shape: Vector(3, 1)
    // arrLabel shape: Vector(3)
    val npoutBack = (npout zip arrLabel.toArray).map { case (data, label) =>
      backward(data, label)
    }
    assert(CheckUtils.reldiff(npoutBack, arrGrad.toArray) < 1e-6f)
  }

  test("regression") {
    checkRegression(Symbol.LogisticRegressionOutput()()(
      Map("data" -> Symbol.Variable("data"), "label" -> Symbol.Variable("label"))),
      (x: Float) => 1.0f / (1.0f + Math.exp(-x).toFloat),
      (x: Float, y: Float) => x - y)
    checkRegression(Symbol.LinearRegressionOutput()()(
      Map("data" -> Symbol.Variable("data"), "label" -> Symbol.Variable("label"))),
      (x: Float) => x,
      (x: Float, y: Float) => x - y)
  }

  // TODO: test softmax

  test("swap axes") {
    val data = Symbol.Variable("data")
    val shape = Shape(2, 3, 4)
    val arrData = NDArray.ones(shape)
    arrData.slice(0).set(1f)
    arrData.slice(1).set(2f)
    // arrData =
    //
    // [[[ 1.,  1.,  1.,  1.],
    //   [ 1.,  1.,  1.,  1.],
    //   [ 1.,  1.,  1.,  1.]],
    //
    // [[ 2.,  2.,  2.,  2.],
    //  [ 2.,  2.,  2.,  2.],
    //  [ 2.,  2.,  2.,  2.]]]
    val swap0 = Symbol.SwapAxis()()(Map("data" -> data, "dim1" -> 0, "dim2" -> 2))
    val swap = Symbol.SwapAxis()()(Map("data" -> swap0, "dim1" -> 1, "dim2" -> 2))
    val exec = swap.bind(Context.cpu(), args = Array(arrData))
    exec.forward()
    val out = exec.outputs(0)

    // After swapaxes(swapaxes(arrData, 0, 2), 1, 2)
    // out should be
    // [[[ 1.,  1.,  1.],
    //   [ 2.,  2.,  2.]],
    //
    //  [[ 1.,  1.,  1.],
    //   [ 2.,  2.,  2.]],
    //
    //  [[ 1.,  1.,  1.],
    //   [ 2.,  2.,  2.]],
    //
    //  [[ 1.,  1.,  1.],
    //   [ 2.,  2.,  2.]]]
    assert(out.shape === Shape(4, 2, 3))
    for (i <- 0 until 4) {
      val axis0 = out.slice(i)
      assert(CheckUtils.reldiff(axis0.toArray, Array(1f, 1f, 1f, 2f, 2f, 2f)) < 1e-6f)
    }
  }

  test("scalar op") {
    val data = Symbol.Variable("data")
    val shape = Shape(3, 4)
    val dataTmp = NDArray.ones(shape) * 5

    val test = {
      import ml.dmlc.mxnet.SymbolConversions._
      2 / (4 - ((1 + data + 1) * 2 / 5) - 0.2)
    }

    val (npout1, npout) = {
      import ml.dmlc.mxnet.NDArrayConversions._
      val npout1 = 4 - ((1 + dataTmp + 1) * 2 / 5) - 0.2f
      val npout = 2 / npout1
      (npout1, npout)
    }

    checkSymbolicForward(test, Array(dataTmp), Array(npout))

    val npoutGrad = new NDArrayConversions(2f * (2f * 2f / 5f)) / (npout1 * npout1)

    checkSymbolicBackward(test, Array(dataTmp), Array(NDArray.ones(shape) * 2), Array(npoutGrad))
  }

  test("scalar pow") {
    val data = Symbol.Variable("data")
    val shape = Shape(1, 1)
    val dataTmp = NDArray.ones(shape) * 3
    val dataTmpPowered = NDArray.ones(shape) * 9
    val test = Symbol.pow(data, 2)
    // TODO: check numeric gradient
    checkSymbolicForward(test, Array(dataTmp), Array(dataTmpPowered))
    checkSymbolicBackward(test, Array(dataTmp), Array(NDArray.ones(shape)), Array(dataTmp * 2))
  }

  test("symbol pow") {
    val shape = Shape(1, 1)

    val data = Symbol.Variable("data")
    val dataTmp = NDArray.ones(shape) * 2

    val exp = Symbol.Variable("exp")
    val expTmp = NDArray.ones(shape) * 3

    val test = Symbol.pow(data, exp)

    // TODO: check numeric gradient
    checkSymbolicForward(test, Seq(dataTmp, expTmp), Seq(NDArray.ones(shape) * 8))

    val dataDir = NDArray.ones(shape) * 4 * expTmp // dataTmp**(expTmp - 1) * expTmp
    // expDir = dataTmp**(expTmp) * log(dataTmp)
    val expDir = NDArray.ones(shape) * 8 * (NDArray.ones(shape) * Math.log(2).toFloat)
    checkSymbolicBackward(test, Seq(dataTmp, expTmp),
                          Seq(NDArray.ones(shape)), Seq(dataDir, expDir))
  }

  test("pow fn") {
    val shape = Shape(3, 4)
    val exp = Symbol.Variable("exp")
    val y = Symbol.pow(2, exp)
    val x = NDArray.ones(shape) * 3
    // TODO: check numeric gradient
    checkSymbolicForward(y, Seq(x), Seq(NDArray.ones(shape) * 8)) // 2**x
    checkSymbolicBackward(y, Seq(x), Seq(NDArray.ones(shape)),
      // log(2) * 2**x
      Seq(NDArray.ones(shape) * 8 * Math.log(2).toFloat))
  }

  test("embedding") {
    val inDim = 10
    val outDim = 4
    val batch = 24

    val data = Symbol.Variable("data")
    val embed = Symbol.Embedding(name = "embed")()(
      Map("data" -> data, "input_dim" -> inDim, "output_dim" -> outDim))
    // TODO
    // scalastyle:off println
    println(s"Embeded symbol: ${embed.toJson}")
    // scalastyle:on println
  }

  // check ops handle duplicate input correctly.
  test("binary op duplicate input") {
    val data = Symbol.Variable("data")
    val shape = Shape(3, 4)
    val dataTmp = NDArray.ones(shape) * 5
    val arrData = dataTmp.copy()
    val arrGrad = NDArray.ones(shape) * 3
    val outGrad = NDArray.ones(shape)
    val square = data * data
    val exeSquare = square.bind(Context.cpu(), args = Array(arrData), argsGrad = Array(arrGrad))
    exeSquare.forward()
    assert(reldiff(exeSquare.outputs.head, dataTmp * dataTmp) < 1e-6f)
    exeSquare.backward(outGrad)
    assert(reldiff(arrGrad, dataTmp * 2f) < 1e-6f)
  }

  test("sign") {
    val data = Symbol.Variable("data")
    val shape = Shape(3, 4)
    val dataTmp = NDArray.ones(shape) * 5
    val arrData = dataTmp.copy()
    val arrGrad = NDArray.ones(shape) * 3

    val test = Symbol.sign()(data)()
    val exeTest = test.bind(Context.cpu(), args = Array(arrData), argsGrad = Array(arrGrad))
    exeTest.forward()
    val out = exeTest.outputs.head
    val npout = NDArray.sign(dataTmp)
    assert(reldiff(out, npout) < 1e-6)

    val outGrad = NDArray.ones(shape) * 2
    exeTest.backward(outGrad)
    arrGrad.toArray.foreach(elem => assert(elem === 0f +- 1e-3f))
  }

  test("round, ceil, floor") {
    val data = Symbol.Variable("data")
    val shape = Shape(3, 4)
    val dataTmp = NDArray.ones(shape) * 5.543f
    val arrData = dataTmp.copy()
    val arrGrad = NDArray.ones(shape) * 2

    val test = Symbol.round()(data)() + Symbol.ceil()(data)() + Symbol.floor()(data)()
    val exeTest = test.bind(Context.cpu(), args = Array(arrData))
    exeTest.forward()
    val out = exeTest.outputs.head
    val npout = NDArray.round(dataTmp) + NDArray.ceil(dataTmp) + NDArray.floor(dataTmp)
    assert(reldiff(out, npout) < 1e-6)
  }

  test("rsqrt, cos, sin") {
    val data = Symbol.Variable("data")
    val shape = Shape(3, 4)
    val dataTmp = NDArray.ones(shape) * 5
    val arrData = dataTmp.copy()
    val arrGrad = NDArray.ones(shape) * 3

    val test = Symbol.rsqrt()(data)() + Symbol.cos()(data)() + Symbol.sin()(data)()
    val exeTest = test.bind(Context.cpu(), args = Array(arrData), argsGrad = Array(arrGrad))
    exeTest.forward()
    val out = exeTest.outputs.head
    val npout = {
      import ml.dmlc.mxnet.NDArrayConversions._
      1 / NDArray.sqrt(dataTmp) + NDArray.cos(dataTmp) + NDArray.sin(dataTmp)
    }
    assert(reldiff(out, npout) < 1e-6)

    val outGrad = NDArray.ones(shape) * 2
    val npoutGrad = {
      import ml.dmlc.mxnet.NDArrayConversions._
      outGrad * -(1 / (2 * dataTmp * NDArray.sqrt(dataTmp))) +
        outGrad * -1 * NDArray.sin(dataTmp) + outGrad * NDArray.cos(dataTmp)
    }
    exeTest.backward(outGrad)
    assert(reldiff(arrGrad, npoutGrad) < 1e-6)
  }

  test("maximum") {
    val data1 = Symbol.Variable("data")
    val data2 = Symbol.Variable("data")
    val shape = Shape(3, 4)
    val dataTmp1 = Random.uniform(0, 100, shape)
    val dataTmp2 = Random.uniform(0, 100, shape)

    val arrData1 = dataTmp1.copy()
    val arrData2 = dataTmp2.copy()

    val test = Symbol.max(data1, data2)
    val exeTest = test.bind(Context.cpu(), args = Array(arrData1, arrData2))
    exeTest.forward()
    val out = exeTest.outputs.head
    val expected = (dataTmp1.toArray zip dataTmp2.toArray).map { case (a, b) => Math.max(a, b) }
    assert(reldiff(out.toArray, expected) < 1e-6)
  }

  test("minimum") {
    val data1 = Symbol.Variable("data")
    val data2 = Symbol.Variable("data")
    val shape = Shape(3, 4)
    val dataTmp1 = Random.uniform(0, 100, shape)
    val dataTmp2 = Random.uniform(0, 100, shape)

    val arrData1 = dataTmp1.copy()
    val arrData2 = dataTmp2.copy()

    val test = Symbol.min(data1, data2)
    val exeTest = test.bind(Context.cpu(), args = Array(arrData1, arrData2))
    exeTest.forward()
    val out = exeTest.outputs.head
    val expected = (dataTmp1.toArray zip dataTmp2.toArray).map { case (a, b) => Math.min(a, b) }
    assert(reldiff(out.toArray, expected) < 1e-6)
  }

  test("transpose") {
    val data = Symbol.Variable("data")
    val test = Symbol.transpose()(data)()

    val shape = Shape(3, 4)
    val ctx = Context.cpu()
    val arrData = Random.uniform(0, 100, shape, ctx)

    val trans: Array[Float] = {
      val tmp = arrData.toArray.toList.grouped(4).toList
      for (i <- 0 until 4) yield {
        List(tmp(0)(i), tmp(1)(i), tmp(2)(i))
      }
    }.flatten.toArray

    val exeTest = test.bind(ctx, args = Map("data" -> arrData))
    exeTest.forward(isTrain = false)
    val out = exeTest.outputs.head

    assert(out.shape == Shape(4, 3))
    assert(reldiff(out.toArray, trans) < 1e-6)
  }

  test("smooth_l1 & makeloss") {
    val data = Symbol.Variable("data")
    val smoothL1 = Symbol.smooth_l1()()(Map("data" -> data, "scalar" -> 1.0f))
    val loss = Symbol.MakeLoss()()(Map("data" -> smoothL1))

    val shape = Shape(2, 6)
    val ctx = Context.cpu()
    val input = NDArray.empty(ctx, shape.toArray: _*)
    val grad = NDArray.empty(ctx, shape.toArray: _*)
    val array = Array[Float](
        -3.5f, -2.5f, -1.5f, -0.5f, -0.3f, -0.1f,
        0.1f, 0.3f, 0.5f, 1.5f, 2.5f, 3.5f)
    input.set(array)

    val arrTmp = Array[Float](
        3.0f, 2.0f, 1.0f, 0.125f, 0.045f, 0.005f,
        0.005f, 0.045f, 0.125f, 1.0f, 2.0f, 3.0f)
    val gradTmp = Array[Float](
        -1.0f, -1.0f, -1.0f, -0.5f, -0.3f, -0.1f,
        0.1f, 0.3f, 0.5f, 1.0f, 1.0f, 1.0f)

    val exeTest =
      loss.bind(ctx, args = Map("data" -> input), argsGrad = Map("data" -> grad))
    exeTest.forward(isTrain = true)
    val out = exeTest.outputs.head

    assert(reldiff(out.toArray, arrTmp) < 1e-6)

    exeTest.backward()

    assert(reldiff(grad.toArray, gradTmp) < 1e-6)
  }

  test("maximum minimum scalar") {
    val data = Symbol.Variable("data")
    val shape = Shape(3, 4)
    val dataTmp = NDArray.ones(shape) * 2

    val arrData = dataTmp.copy()

    val test = Symbol.max(data, 3) + Symbol.max(9, data) + Symbol.min(5, data) + Symbol.min(data, 4)
    val exeTest = test.bind(Context.cpu(), args = Array(arrData))
    exeTest.forward()
    val out = exeTest.outputs.head
    // 3 + 9 + 2 + 2
    assert(reldiff(out, NDArray.ones(shape) * 16) < 1e-6)
  }

  test("abs") {
    val data = Symbol.Variable("data")
    val shape = Shape(3, 4)
    val dataTmp = NDArray.ones(shape) * 5
    val arrData = dataTmp.copy()
    val arrGrad = NDArray.ones(shape) * 3

    val test = Symbol.abs()(data)()
    val exeTest = test.bind(Context.cpu(), args = Array(arrData), argsGrad = Array(arrGrad))
    exeTest.forward()
    val out = exeTest.outputs.head
    val npout = NDArray.abs(dataTmp)
    assert(reldiff(out, npout) < 1e-6)

    val outGrad = NDArray.ones(shape) * 2
    val npoutGrad = outGrad * NDArray.sign(dataTmp)
    exeTest.backward(outGrad)
    assert(reldiff(arrGrad, npoutGrad) < 1e-6)
  }

  // configure A: input --> conv --> deconv --> output.
  // the convolution and deconvoluiton has similar parameter which ensure
  // the input shape is the same as output, and the same weights between conv
  // and deconv;
  // If the input value of forward() and backwrad() is the same, then
  // the output value of them should also the same;
  private def checkDeconvolutionForwardBackward(inputShape: Shape,
                                                numFilter: Int,
                                                kernel: (Int, Int),
                                                stride: (Int, Int),
                                                pad: (Int, Int)): Unit = {
    require(inputShape(1) == numFilter)
    val data = Sy