import joblib
import pandas
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier


from constants import NAIVE_BAYES_ID, SVM_ID, DATA_KEY, PREPROCESSED_DATA_FILE, TARGET_KEY, N_FOLDS
MODEL_COMPRESS_LEVEL = 5


def get_test_file_name(fold):
    return "./test_data/test_{}.csv".format(fold)


def get_model_file_name(algorithm_id, fold):
    return "./model_data/{}_{}.model".format(algorithm_id, fold)


def get_vectorizer_file_name(fold):
    return "./vectorizer_data/vectorizer_{}.model".format(fold)


def train_naive_bayes(data, target, fold):
    multinomial_nb = OneVsRestClassifier(MultinomialNB())
    multinomial_nb.fit(data, target)
    joblib.dump(multinomial_nb, get_model_file_name(NAIVE_BAYES_ID, fold), compress=MODEL_COMPRESS_LEVEL)
    pass


def train_svm(data, target, fold):
    support_vector_machine = OneVsRestClassifier(svm.SVC())
    support_vector_machine.fit(data, target)
    joblib.dump(support_vector_machine, get_model_file_name(SVM_ID, fold), compress=MODEL_COMPRESS_LEVEL)
    pass


if __name__ == "__main__":
    data_frame = pandas.read_csv(PREPROCESSED_DATA_FILE)

    data = data_frame[DATA_KEY].to_numpy()
    target = data_frame[TARGET_KEY].to_numpy()


    kFolder = KFold(n_splits=N_FOLDS)
    fold_count = 0

    for train_index, test_index in kFolder.split(data):
        data_train, target_train = data[train_index], target[train_index]

        tfidf_vectorizer = TfidfVectorizer(
            stop_words=stopwords.words('indonesian')
        )

        processed_data_train = tfidf_vectorizer.fit_transform(
            data_train
        ).toarray()

        joblib.dump(
            tfidf_vectorizer,
            get_vectorizer_file_name(fold_count)
        )

        train_naive_bayes(processed_data_train, target_train, fold_count)
        train_svm(processed_data_train, target_train, fold_count)

        test_data = pandas.DataFrame({
            DATA_KEY: data[test_index],
            TARGET_KEY: target[test_index],
        })

        test_data.to_csv(get_test_file_name(fold_count))
        fold_count += 1
