import joblib
from numpy.lib.function_base import average
import pandas
from nltk.corpus import stopwords
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
from pycm import ConfusionMatrix
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean

def draw_conf_matrix(confusion_matrix: ConfusionMatrix, algo_name: str, fold_iter: int):
    classification_classes = confusion_matrix.classes
    confusion_matrix_df = DataFrame(confusion_matrix.to_array())
    row_sums = confusion_matrix_df.sum(axis=1).to_list()

    x_labels = ["{} ({})".format(label, row_sums[index])
                for index, label in enumerate(classification_classes)]

    temp_df = confusion_matrix_df.T

    shape = temp_df.shape
    annotations = [
        [f'''{classification_classes[row]}\n{classification_classes[col]}\n{temp_df[row][col]}'''
            for col in range(0, shape[0])]
        for row in range(0, shape[1])
    ]

    confusion_matrix_heatmap = sns.heatmap(
        confusion_matrix_df.T,
        annot=DataFrame(annotations).T,
        xticklabels=x_labels,
        yticklabels=classification_classes,
        cmap="OrRd",
        linewidths=0.1,
        linecolor="black",
        fmt='',
        cbar=False,
    )

    fig = confusion_matrix_heatmap.get_figure()
    fig.savefig(
        "./test_result_data/{}-fold-{}.svg".format(algo_name, fold_iter),
        bbox_inches='tight'
    )
    fig.savefig(
        "./test_result_data/{}-fold-{}.png".format(algo_name, fold_iter),
        bbox_inches='tight'
    )
    plt.clf()
    pass


sns.set(
    font="Monospace",
)

warnings.filterwarnings("ignore")

from constants import NAIVE_BAYES_ID, SVM_ID, DATA_KEY, TARGET_KEY, N_FOLDS, ALGORITHM_LABELS
from train import get_model_file_name, get_test_file_name, get_vectorizer_file_name


def get_test_result_file_name(fold):
    return "./test_result_data/hasil_uji_{}.csv".format(fold)


tf_idf_converter = TfidfVectorizer(
    stop_words=stopwords.words('indonesian')
)

ALGORITHMS = [
    NAIVE_BAYES_ID,
    SVM_ID
]

test_results_per_algorithm = {
    NAIVE_BAYES_ID: [],
    SVM_ID: [],
}

class_labels = ["komputasi", "SI", "Jaringan"]
class_labels_map = {class_label: index for index, class_label in enumerate(class_labels)}

report_text_file = open("./test_result_data/hasil_perhitungan.txt", "w")

def divide(p, q, coalesce = 0):
    if q == 0:
        return coalesce
    else:
        return p / q

test_results_summary = []

for fold in range(0, N_FOLDS):
    fold_iteration = fold + 1
    test_results_per_fold = []

    for algorithm_id in ALGORITHMS:
        print("Processing algorithm {}, fold {}".format(
            ALGORITHM_LABELS[algorithm_id],
            fold_iteration,
        ))

        model = joblib.load(
            get_model_file_name(algorithm_id, fold)
        )

        test_file = pandas.read_csv(get_test_file_name(fold))
        data_test = test_file[DATA_KEY]
        target_test = test_file[TARGET_KEY]

        tfidf_vectorizer = joblib.load(
            get_vectorizer_file_name(fold)
        )

        processed_data_test = tfidf_vectorizer.transform(
            data_test
        ).toarray()

        predicted_data_test = model.predict(processed_data_test)

        precision, recall, f_score, support = precision_recall_fscore_support(
            target_test,
            predicted_data_test,
            average='macro'
        )

        report_confusion_matrix = ConfusionMatrix(
            actual_vector=target_test.to_list(),
            predict_vector=predicted_data_test
        )

        draw_conf_matrix(
            report_confusion_matrix,
            ALGORITHM_LABELS[algorithm_id],
            fold_iteration,
        )

        positions = report_confusion_matrix.position()
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []

        report_text = ""

        for classname in report_confusion_matrix.classes:
            tp = len(positions[classname]["TP"])
            tn = len(positions[classname]["TN"])
            fp = len(positions[classname]["FP"])
            fn = len(positions[classname]["FN"])

            temp_recall = divide(tp, tp + fn)
            temp_precision = divide(tp, tp + fp)
            temp_f1_score = divide(2 * tp, 2 * tp + fp + fn)
            temp_accuracy = divide(tp + tn, tp + tn + fp + fn)

            recalls.append(temp_recall)
            precisions.append(temp_precision)
            f1_scores.append(temp_f1_score)
            accuracies.append(temp_accuracy)

            report_text += f'''
Untuk pengujian pada fold ke-{fold_iteration} dari algoritma {ALGORITHM_LABELS[algorithm_id]}, kelas '{classname}' memiliki nilai true positive (tp) = {tp:d}, true negative (tn) = {tn:d}, false positive (fp) = {fp:d}, false negative (fn) = {fn:d}. Nilai recall = tp / tp + fn = {temp_recall:.4f}, nilai precision = tp / tp + fp = {temp_precision:.4f}, accuracy = tp + tn / tp + tn + fp + fn = {temp_accuracy:.4f}.
'''.strip()
            pass

        report_text += f'''
Rata-rata dari seluruh nilai precision adalah {precision:.4f}. \
Rata-rata dari seluruh nilai recall adalah {recall:.4f}. \
Rata-rata dari seluruh nilai f1-score adalah {f_score:.4f}. \
Nilai accuracy terkecil dari seluruh nilai adalah {(min(accuracies)):.4f}.
'''.strip()

        print(report_text, file=report_text_file, end="\n\n")

        accuracy = accuracy_score(target_test, predicted_data_test)

        test_results_per_algorithm[algorithm_id].append({
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f_score,
            "Accuracy": accuracy,
        })

        test_results_per_fold.append({
            "Algoritma": ALGORITHM_LABELS[algorithm_id],
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f_score,
            "Accuracy": accuracy,
        })

        test_results_summary.append({
            "Algoritma / Fold": "{} / {}".format(ALGORITHM_LABELS[algorithm_id], fold_iteration),
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f_score,
            "Accuracy": accuracy,
        })

    pandas.DataFrame(
        test_results_per_fold,
    ).to_csv(
        get_test_result_file_name(fold + 1)
    )

summary_df = DataFrame(test_results_summary)
summary_df.to_excel("./test_summary.xlsx")

for algorithm_id, test_result in test_results_per_algorithm.items():
    data_frame = pandas.DataFrame(
        test_result
    )

    mean = data_frame.mean()

    mean.to_csv(
        get_test_result_file_name(
            "Rata-Rata " + ALGORITHM_LABELS[algorithm_id]
        )
    )
