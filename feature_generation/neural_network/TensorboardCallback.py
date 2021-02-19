import io
import os
import warnings
from hashlib import md5
from tempfile import gettempdir

from google.cloud import storage
from keras import backend as K
from keras.callbacks import TensorBoard

try:
    import tensorflow as tf
    from tensorboard.plugins import projector
except ImportError:
    raise ImportError("You need the TensorFlow module installed to " "use TensorBoard.")


class BucketTensorBoard(TensorBoard):
    """TensorBoard basic visualizations, but each callback is sent to a
    cloud bucket.
    [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=gs://your-bucket-uri/some_sub_directory/
    ```
    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.
    # Arguments
        bucket_uri: the uri of the bucket where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved. If set to 0, embeddings won't be computed.
            Data to be visualized in TensorBoard's Embedding tab must be passed
            as `embeddings_data`.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/guide/embedding#metadata)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
        embeddings_data: data to be embedded at layers specified in
            `embeddings_layer_names`. Numpy array (if the model has a single
            input) or list of Numpy arrays (if the model has multiple inputs).
            Learn [more about embeddings](
            https://www.tensorflow.org/guide/embedding).
        update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
            the losses and metrics to TensorBoard after each batch. The same
            applies for `'epoch'`. If using an integer, let's say `10000`,
            the callback will write the metrics and losses to TensorBoard every
            10000 samples. Note that writing too frequently to TensorBoard
            can slow down your training.
    """

    def __init__(
        self,
        bucket_uri,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq="epoch",
    ):
        super(BucketTensorBoard, self).__init__(
            histogram_freq=histogram_freq,
            batch_size=batch_size,
            write_graph=write_graph,
            write_grads=write_grads,
            write_images=write_images,
            embeddings_freq=embeddings_freq,
            embeddings_layer_names=embeddings_layer_names,
            embeddings_metadata=embeddings_metadata,
            embeddings_data=embeddings_data,
            update_freq=update_freq,
        )

        # Parses the bucket_uri
        if bucket_uri.startswith("gs://"):
            self.bucket_protocol = "gs"
            split_bucket_name = bucket_uri[5:].split("/", 1)
            if len(split_bucket_name) == 1:
                bucket_name = split_bucket_name[0]
                self.extra_path = ""
            else:
                bucket_name = split_bucket_name[0]
                self.extra_path = split_bucket_name[1].strip("/")

            # Sets up the GCP client and gets the bucket
            try:
                self.bucket = storage.Client().get_bucket(bucket_name)
            except:
                raise Exception(
                    f'Bucket with name "{bucket_name}" could not be fetched'
                )

            # Gets the list of current blobs, so it doesn't upload them twice
            self.current_blobs = {}
            for item in self.bucket.list_blobs(prefix=self.extra_path):
                self.current_blobs[item.name] = item.metadata["md5_hash"]
        else:
            raise Exception("The protocol informed in the URI is not supported")

        self.log_dir = (
            f"{gettempdir()}/tensorboard_callbacks/{bucket_name}/{self.extra_path}"
        )

    def on_epoch_end(self, logs, index):
        super(BucketTensorBoard, self).on_epoch_end(logs, index)
        try:
            self._write_logs_gs()
        except:
            print("could not upload files to gcs")

    def _write_logs_gs(self):
        train_dir = [
            os.path.join("train", file)
            for file in os.listdir(os.path.join(self.log_dir, "train"))
        ]
        validation_dir = [
            os.path.join("validation", file)
            for file in os.listdir(os.path.join(self.log_dir, "validation"))
        ]
        tensorboard_files = [
            os.path.join(self.log_dir, file) for file in validation_dir + train_dir
        ]
        files_to_upload = filter(os.path.isfile, tensorboard_files)
        for file_path in list(files_to_upload):
            file_name = os.path.join(
                "tensorboard", *file_path.split("/")[-2:]
            )  # most unreadable code 2020
            log_file = open(file_path, "rb")
            md5_hash = md5(log_file.read()).hexdigest()
            log_file.close()
            if (file_name not in self.current_blobs) or (
                self.current_blobs[file_name] != md5_hash
            ):
                blob = self.bucket.blob(f"{self.extra_path}/{file_name}")
                blob.metadata = {"md5_hash": md5_hash}
                blob.upload_from_filename(file_path)
                self.current_blobs[file_name] = md5_hash