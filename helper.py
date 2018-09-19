import numpy as np
import scipy
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append('/media/proto/E490-E3B6/IA/Reinforcement/ABN_Robotics/surfer/meinemashine/Meta-Relational-A3C')

print(sys.path)

def set_image_context(correct, observation,values,selection,trial):
    obs = observation * 225.0
    obs_a = obs[:,0:1,:]
    obs_b = obs[:,1:2,:]
    cor = correct * 225.0
    obs_a = scipy.misc.imresize(obs_a,[100,100],interp='nearest')
    obs_b = scipy.misc.imresize(obs_b,[100,100],interp='nearest')
    cor = scipy.misc.imresize(cor,[100,100],interp='nearest')
    bandit_image = Image.open('/media/proto/E490-E3B6/IA/Reinforcement/ABN_Robotics/surfer/meinemashine/Meta-Relational-A3C/resources/c_bandit.png')
    draw = ImageDraw.Draw(bandit_image)
    font = ImageFont.truetype("/media/proto/E490-E3B6/IA/Reinforcement/ABN_Robotics/surfer/meinemashine/Meta-Relational-A3C/resources/FreeSans.ttf", 24)
    draw.text((50, 360),'Trial: ' + str(trial),(0,0,0),font=font)
    draw.text((50, 330),'Reward: ' + str(values),(0,0,0),font=font)
    bandit_image = np.array(bandit_image)
    bandit_image[120:220,0:100,:] = obs_a
    bandit_image[120:220,100:200,:] = obs_b
    bandit_image[0:100,50:150,:] = cor
    bandit_image[291:297,10+(selection*95):10+(selection*95)+80,:] = [80.0,80.0,225.0]
    return bandit_image


#This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy
  
  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)
  
  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration,verbose=False)