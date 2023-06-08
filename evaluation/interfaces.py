import os

# https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_/ijb
class Mxnet_model_interf:
    def __init__(self, model_file, layer="fc1", image_size=(112, 112)):
        import mxnet as mx

        self.mx = mx
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if len(cvd) > 0 and int(cvd) != -1:
            ctx = [self.mx.gpu(ii) for ii in range(len(cvd.split(",")))]
        else:
            ctx = [self.mx.cpu()]

        prefix, epoch = model_file.split(",")
        print(">>>> loading mxnet model:", prefix, epoch, ctx)
        sym, arg_params, aux_params = self.mx.model.load_checkpoint(prefix, int(epoch))
        all_layers = sym.get_internals()
        sym = all_layers[layer + "_output"]
        model = self.mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[("data", (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.transpose(0, 3, 1, 2)
        data = self.mx.nd.array(imgs)
        db = self.mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        emb = self.model.get_outputs()[0].asnumpy()
        return emb


class Torch_model_interf:
    def __init__(self, model_file, image_size=(112, 112)):
        import torch

        self.torch = torch
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        device_name = "cuda:0" if len(cvd) > 0 and int(cvd) != -1 else "cpu"
        self.device = self.torch.device(device_name)
        try:
            self.model = self.torch.jit.load(model_file, map_location=device_name)
        except:
            print(
                "Error: %s is weights only, please load and save the entire model by `torch.jit.save`"
                % model_file
            )
            self.model = None

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.transpose(0, 3, 1, 2).copy().astype("float32")
        imgs = (imgs - 127.5) * 0.0078125
        output = self.model(self.torch.from_numpy(imgs).to(self.device).float())
        return output.cpu().detach().numpy()


class ONNX_model_interf:
    def __init__(self, model_file, image_size=(112, 112)):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        self.ort_session = ort.InferenceSession(
            model_file, providers=["CUDAExecutionProvider"]
        )
        print(self.ort_session.get_providers())
        exit()
        self.output_names = [self.ort_session.get_outputs()[0].name]
        self.input_name = self.ort_session.get_inputs()[0].name

    def __call__(self, imgs):
        imgs = imgs.transpose(0, 3, 1, 2).astype("float32")
        imgs = (imgs - 127.5) * 0.0078125
        outputs = self.ort_session.run(self.output_names, {self.input_name: imgs})
        return outputs[0]


def keras_model_interf(model_file):
    import tensorflow as tf
    from tensorflow_addons.layers import StochasticDepth

    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    mm = tf.keras.models.load_model(model_file, compile=False)
    return lambda imgs: mm((tf.cast(imgs, "float32") - 127.5) * 0.0078125).numpy()