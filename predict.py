from cog import BaseModel, Input, Path
from model_train import train

MODEL_NAME = "./jugger.safetensors"
MODEL_CACHE = "model-cache"


class Predictor(BaseModel):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

    def predict(
        self,
        input_images: Path = Input(
            description="A .zip or .tar file containing the image files that will be used for fine-tuning"
        ),
    ) -> Path:
        training_result = train(
            input_images,
            None,
            1024,
            1,
            160,
            True,
            1e-6,
            3e-4,
            1e-4,
            32,
            "constant",
            100,
            "TOK",
            "a photo of TOK, ",
            None,
            False,
            True,
            1.0,
            True,
            999999,
            "infer",
        )
        return training_result.weights
