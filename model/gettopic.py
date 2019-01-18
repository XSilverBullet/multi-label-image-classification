
import numpy as np
import lda

def get_train_test_lda( y_train, y_test, topic):


    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')
    for i in range(len(topic)):

        global x1, x2
        model_label = lda.LDA(n_topics=topic[i], n_iter=1000)
        model_label.fit(y_train)

        doc_topic = model_label.doc_topic_
        test_doc_topic = model_label.transform(y_test)
        if i<=0:
            x1 = np.array(doc_topic)
            x2 = np.array(test_doc_topic)

        else:
            x1 = np.hstack((doc_topic, x1))
            x2 = np.hstack((test_doc_topic, x2))

    return np.array(x1), np.array(x2)
