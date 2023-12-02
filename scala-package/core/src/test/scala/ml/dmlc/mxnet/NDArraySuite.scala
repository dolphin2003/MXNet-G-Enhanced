package ml.dmlc.mxnet

import java.io.File
import java.util.concurrent.atomic.AtomicInteger

import ml.dmlc.mxnet.NDArrayConversions._
import org.scalatest.{Matchers, BeforeAndAfterAll, FunSuite}

class NDArraySuite extends FunSuite with BeforeAndAfterAll with Matchers {
  private val sequence: AtomicInteger = new AtomicInteger(0)

  test("to java array") {
    val ndarray = NDArray.zeros(2, 2)
    assert(ndarray.toArray === Array(0f, 0f, 0f, 0f))
  }

  test("to scalar") {
    val ndzeros = NDArray.zeros(1)
    assert(ndzeros.toScalar === 0f)
    val ndones = NDArray.ones(1)
    assert(ndones.toScalar === 1f)
  }

  test ("call toScalar on an ndarray which is not a scalar") {
    intercept[Exception] { NDArray.zeros(1, 1).toScalar }
  }

  test("size and shape") {
    val ndzeros = NDArray.zeros(4, 1)
    assert(ndzeros.shape === Shape(4, 1))
    assert(ndzeros.size === 4)
  }

  test("set scalar value") {
    val ndarray = NDArray.empty(2, 1)
    ndarray.set(10f)
    assert(ndarray.toArray === Array(10f, 10f))
  }

  test("copy from java array") {
    val ndarray = NDArray.empty(4, 1)
    ndarray.set(A