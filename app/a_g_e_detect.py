import numpy as np
import matplotlib.pyplot as plt
import sys


from facelib import AgeGenderEstimator, EmotionDetector
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D




def get_age_gender_emotion(faces,age_gender_detector, emotion_detector):
    genders, ages = age_gender_detector.detect(faces)
    list_of_emotions, probab = emotion_detector.detect_emotion(faces,verbose=True)

    probabilities = []
    for prob in probab:
        probabilities.append(prob.exp().detach().numpy()/prob.exp().sum().detach().numpy())
    
    return genders, ages, list_of_emotions, probabilities
    


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta



def create_images_age_gender_emotion(faces, ages, genders, emotions, probs):
    
    N=emotions.shape[0]
    theta = radar_factory(N, frame ='polygon')
    c="r"
    plt.rc('font', size=6) #controls default text size
    px = 1/plt.rcParams['figure.dpi']
    fig, axes = plt.subplots(figsize=(5*faces.shape[0]*100*px, 400*px),ncols=faces.shape[0], nrows=1,subplot_kw=dict(projection='radar'))

    try:
        _ = axes[0]
    except:
        axes = [axes] 

    for i,probs in enumerate(probs):
        _=axes[i].set_title(f"Gender:{genders[i]}" + " "*4 + f"Age:{ages[i]}")
        _=axes[i].tick_params(labelleft=False)
        _=axes[i].plot(theta, probs, color=c)
        _=axes[i].fill(theta, probs,facecolor=c, alpha=0.5)
        _=axes[i].set_varlabels(emotions)

    plt.savefig(sys.path[0]+"/processed/age_gender_emotion/estimations.svg",dpi=150)

