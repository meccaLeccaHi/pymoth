def extractMNISTFeatureArray( mnist, labels, image_indices, phase_label ):
    # Extract a subset of the samples from each class, convert the images to doubles on [0 1], and
    #     return a 4-D array: 1, 2 = im. 3 indexes images within a class, 4 is the class.
    #
    # Inputs:
    #   mnist = struct loaded by 'load mnistAll_plusSubsets'
    #      with fields = training_images, test_images, training_labels, test_labels
    #      trI = mnist.train_images;
    #      teI = mnist.test_images;
    #      trL = mnist.train_labels;
    #      teL = mnist.test_labels;
    #   labels = vector of the classes (digits) you want to extract
    #   image_indices = list of which images you want from each class
    #   phase_label = 'train' or 'test'. Determines which images you draw from
    #      (since we only need a small subset, one or the other is fine)
    #
    # Outputs:
    #   im_array = numberImages x h x w x numberClasses 4-D array

    import numpy as np

    # get some dimensions:
    (h,w) = mnist['train_images'].shape[1:3]
    max_ind = max(image_indices)

    # initialize outputs:
    im_array = np.zeros((max_ind+1, h, w, len(labels)))

    # process each class in turn:
    for c in labels:
        if phase_label=='train': # 1 = extract train, 0 = extract test
            im_data = mnist['train_images']
            target_data = mnist['train_labels']
        else:
            im_data = mnist['test_images']
            target_data = mnist['test_labels']

        # Convert to double precision float (https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html)
        class_array = im_data[np.where(target_data==c)].astype('float64')/256

        im_array[image_indices,:,:,c] = class_array[image_indices,:,:]

    return im_array
