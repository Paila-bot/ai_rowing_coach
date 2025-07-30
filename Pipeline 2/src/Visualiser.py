from PIL import ImageDraw
class Visualizer:
    @staticmethod
    def draw_pose(img, joints):
        draw = ImageDraw.Draw(img)
        pairs = [("head", "shoulder_left"), ("head", "shoulder_right"),
                 ("shoulder_left", "hips"), ("shoulder_right", "hips"),
                 ("hips", "knees")]
        for a, b in pairs:
            if a in joints and b in joints:
                draw.line([joints[a][::-1], joints[b][::-1]], fill='green', width=5)
        return img