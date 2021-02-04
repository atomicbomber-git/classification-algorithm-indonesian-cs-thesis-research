<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport"
          content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>
        Klasifikasi Jurnal Informatika
    </title>

    <link rel="stylesheet" href="assets/app.css">


</head>
<body>
<div class="container my-4">
    <h1> Klasifikasi Jurnal Informatika </h1>
    <hr>

    <fieldset>
        <legend class="font-weight-bold text-primary text-uppercase">
            Klasifikasikan Dokumen
        </legend>
        <form method="POST" enctype="multipart/form-data" action="process">
            <div class="form-group">
                <label for="document"> Berkas Dokumen </label>
                <input
                        name="document"
                        class="form-control"
                        id="document"
                        type="file">
            </div>

            <h2> Algoritma </h2>

            <div class="form-group">
                 % for algorithm in algorithms:
                <div>
                    <input
                        value="{{ algorithm["id"] }}-{{ algorithm["fold"] }}"
                        name="algorithm"
                        id="algorithm_{{ algorithm["id"] }}_{{ algorithm["fold"] }}" type="radio">

                    <label for="algorithm_{{ algorithm["id"] }}_{{ algorithm["fold"] }}">
                        {{ algorithm["label"] }} (Fold {{ algorithm["fold"] + 1 }})
                    </label>
                % end
                </div>
            </div>

            <div class="d-flex justify-content-end">
                <button class="btn btn-primary">
                    Klasifikasikan
                </button>
            </div>
        </form>
    </fieldset>

    % if message is not None:
        <div class="alert alert-info">
            {{ message }}
        </div>
    % end
</div>

<script src="assets/app.js"></script>
</body>
</html>