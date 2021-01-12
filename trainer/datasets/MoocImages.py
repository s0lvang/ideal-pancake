from trainer.configuration.Heatmap import Heatmap


class MoocImages(Heatmap):
    def __init__(self):
        super().__init__("mooc-images")
        self.subject_id_column = "subject"
        self.label = "posttest"

    def __str__(self):
        return super().__str__()
