# Playing around with MLEM

MLEM promises that you:

> Use the same human-readable format for any ML framework

This is a bold promise and in this repo I will explore it a little (and maybe some other features as well).
Note that the machine learning part of the content is only secondary.
In the foreground we put the process and the tools.

## Fetching and preparing the data 👷🏽‍♀️

To keep it simple on the ML front, we use the [Iris data set](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).
The data is obtained in [`get_data.py`](./get_data.py); see the comments there for more details.

This script is used in the first stage of the DVC pipeline which is coded in [`dvc.yaml`](./dvc.yaml).

## Training the model and persisting it using MLEM 🚀

In [`train_and_persist.py`](./train_and_persist.py) we, well, train and persist the model.
Again, in `dvc.yaml` this script is used as the second stage.
Here's it is important to pay more attention to the `mlem.api.save()` statement:

```python
save(
    rf, "rf", sample_data=X, description="Random Forest Classifier",
)
```

`rf` is the fitted model and it is given a name; the _string_ `"rf"`.
In addition a description is provided.
Furthermore, by providing a value to the parameter `sample_data`, MLEM will include the schema of the data in the model's meta data.
Checkout [`.mlem/model/rf.mlem`](./.mlem/model/rf.mlem).

## What's next? Or how to get predictions using an API? ⚡️

By running `dvc repro` in this project following things will happen:

- Iris data set will be fetched and splitted into train and test sets.
- A model will be train.
- The model will be persisted by MLEM; its metadata ([`.mlem/model/rf.mlem`](./.mlem/model/rf.mlem)) will be tracked by Git and the model itself ([`.mlem/model/rf`](./.mlem/model/rf)) will be tracked by DVC.

Now comes the fun part.
By running:

```bash
mlem build rf docker --conf server.type=fastapi --conf image.name=rf-image-test
```

MLEM will build a docker image that can be used to get predictions from the trained model using RESTful API.
Once the image is built, a container can be ran:

```
docker run --rm -it -p 8080:8080 rf-image-test
```

Once it is up and running, the documentation of the endpoints of the new API can be found here: http://0.0.0.0:8080/docs.

To make it easier, [`Taskfile.yml`](./Taskfile.yml) can help in building and serving the image.
See [`task`](https://taskfile.dev/) for more details.

**TBA: Evaluate the model by getting predictions from the API for the test set.**
