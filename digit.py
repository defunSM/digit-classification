import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn import svm

def main():
    labeled_images = pd.read_csv('train.csv')
    images = labeled_images.iloc[0:10000, 1:]
    labels = labeled_images.iloc[0:10000, :1]
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

    test_images[test_images>0]=1
    train_images[train_images>0]=1

    i=2
    img=train_images.iloc[i].as_matrix()
    img=img.reshape((28,28))
    plt.imshow(img,cmap='binary')
    plt.title(train_labels.iloc[i,0])
    plt.hist(train_images.iloc[i])

    plt.show()

    clf = svm.SVC(class_weight='balanced', max_iter=1000)
    clf.fit(train_images, train_labels.values.ravel())
    print(clf.score(test_images, test_labels))



if __name__ == "__main__":
    main()
