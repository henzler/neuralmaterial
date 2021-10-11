from lib.core.logger import CoreLogger

class Logger(CoreLogger):
    def __init__(self):
        super().__init__()

    def log_images(self, image_dict, mode, step):

        for k, v in image_dict.items():
            if 'normal' in k:
                v = (v + 1) / 2

            self.writer.add_images(f'{mode}_{k}', v, step)

    def log(self, outputs, mode, step):
        self.log_images(outputs['images'], mode, step)