from trainer.configuration.Heatmap import Heatmap


class EMIPImages(Heatmap):
    def __init__(self):
        super().__init__("emip-images")
        self.subject_id_column = "id"
        self.label = "age"

    def __str__(self):
        return super().__str__()