/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_predict_api.h
 * \brief C predict API of mxnet, contains a minimum API to run prediction.
 *  This file is self-contained, and do not dependent on any other files.
 */
#ifndef MXNET_C_PREDICT_API_H_
#define MXNET_C_PREDICT_API_H_

#ifdef __cplusplus
#define MXNET_EXTERN_C extern "C"
#else
#define MXNET_EXTERN_C
#endif

#ifdef _WIN32
#ifdef MXNET_EXPORTS
#define MXNET_DLL MXNET_EXTERN_C __declspec(dllexport)
#else
#define MXNET_DLL MXNET_EXTERN_C __declspec(dllimport)
#endif
#else
#define MXNET_DLL MXNET_EXTERN_C
#endif

/*! \brief manually define unsigned int */
typedef unsigned int mx_uint;
/*! \brief manually define float */
typedef float mx_float;
/*! \brief handle to Predictor */
typedef void *PredictorHandle;
/*! \brief handle to NDArray list */
typedef void *NDListHandle;

/*!
 * \brief Get the last error happeneed.
 * \return The last error happened at the predictor.
 */
MXNET_DLL const char* MXGetLastError();

/*!
 * \brief create a predictor
 * \param symbol_json_str The JSON string of the symbol.
 * \param param_bytes The in-memory raw bytes of parameter ndarray file.
 * \param param_size The size of parameter ndarray file.
 * \param dev_type The device type, 1: cpu, 2:gpu
 * \param dev_id The device id of the predictor.
 * \param num_input_nodes Number of input nodes to the net,
 *    For feedforward net, this is 1.
 * \param input_keys The name of input argument.
 *    For feedforward net, this is {"data"}
 * \param input_shape_indptr Index pointer of shapes of each input node.
 *    The length of this array = num_input_nodes + 1.
 *    For feedforward net that takes 4 dimensional input, this is {0, 4}.
 * \param input_shape_data A flatted data of shapes of each input node.
 *    For feedforward net that takes 4 dimensional input, this is the shape data.
 * \param out The created predictor handle.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredCreate(const char* symbol_json_str,
                           const void* param_bytes,
                           int param_size,
                           int dev_type, int dev_id,
                           mx_uint num_input_nodes,
                           const char** input_keys,
                           const mx_uint* input_shape_indptr,
                           const mx_uint* input_shape_data,
                           PredictorHandle* out);

/*!
 * \brief create a predictor wich customized outputs
 * \param symbol_json_str The JSON string of the symbol.
 * \param param_bytes The in-memory raw bytes of parameter ndarray file.
 * \param param_size The size of parameter ndarray file.
 * \param dev_type The device type,