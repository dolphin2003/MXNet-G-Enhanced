package org.dmlc.mxnet;

import android.graphics.Bitmap;
import android.graphics.Color;

public class Predictor {
  static {
    System.loadLibrary("mxnet_predict");
  }

  public static class InputNode {
    String key;
    int[] shape;
	public InputNode(String key, int[] shape) {
		this.key = key;
		this.shape = shape;
	}
  }

  public static class Device {
    public enum Type {
      CPU, GPU, CPU_PINNED
    }

	public Device(Type t, int i) {
		this.type = t;
		this.id = i;
	}

    Type type;
    int id;
    int ctype() {
      return this.type == Type.CPU? 1: this.type == Type.GPU? 2: 3;
    }
  }

  private long handle = 0;

  public Predictor(byte[] symbol, byte[] params, Device dev, InputNode[] input) {
	String[] keys = new String[input.length]; 
	int[][] shapes = new int[input.length][];
	for (int i=0; i<input.length; ++i) {
		keys[i] = input[i].key;
		shapes[i] = input[i].shape;
	}
    this.handle = createPredictor(symbol, params, dev.ctype(), dev.id, keys, shapes);
  }

  public void free() {
    if (this.handle != 0) {
      nativeFree(handle);
      this.handle = 0;
    }
  }

  public float[] getOutput(int index) {
    if (this.handle == 0) return null;
    return nativeGetOutput(this.handle, index);
  }


  public void forward(String key, float[] input) {
      if (this.handle == 0) return;
      nativeForward(this.handle, key, input);
  }

  static public float[] inputFromImage(Bitmap[] bmps, float meanR, float meanG, float meanB) {
    if (bmps.length == 0) return null;

    int width = bmps[0].getWidth();
    int height = bmps[0].getHeight();
    float[] buf = new float[height * width * 3 * bmps.length];
    for (int x=0; x<bmps.length; x++) {
  