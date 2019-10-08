from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = load_img('B_0_0_0_0_0.png')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator

#datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
#datagen = ImageDataGenerator(rotation_range=10)
datagen = ImageDataGenerator(rotation_range=30)

# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot

for i in range(12):
	# define subplot
	#pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	img_array = img_to_array(image)
	# plot raw pixel data
	#pyplot.imshow(image)
	save_img(str(i) + '.jpg', img_array)
# show the figure
pyplot.show()