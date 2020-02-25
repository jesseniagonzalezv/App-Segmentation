var el = x => document.getElementById(x);

function showPicker(inputId) { el('file-input').click(); }

function showPicked(input) {
    el('upload-label').innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function (e) {
        el('image-picked').src = e.target.result;
        el('image-picked').className = '';
    }
    reader.readAsDataURL(input.files[0]);
}

//function saveBlob(blob, fileName) {
//    var a = document.createElement('a');
//    a.href = window.URL.createObjectURL(blob);
//    a.download = fileName;
//    a.dispatchEvent(new MouseEvent('click'));
//}

function analyze() {
    var uploadFiles = el('file-input').files;
    if (uploadFiles.length != 1) alert('Please select 1 file to analyze!');

    el('analyze-button').innerHTML = 'Analyzing...';
    var xhr = new XMLHttpRequest();
    var loc = window.location
    xhr.open('POST', `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`, true);
    xhr.onerror = function() {alert (xhr.responseText);}
    xhr.responseType = 'blob' // We are expecting a file back (csv with predictions)
    xhr.onload = function(e){
        if (this.status == 200){
        // Create a new Blob object using the response data of the onload object
        var blob = new Blob([this.response], {type: 'text/csv'});
        //Create a link element, hide it, direct it towards the blob, and then 'click' it programatically
        let a = document.createElement("a");
        a.style = "display: none";
        document.body.appendChild(a);
        //Create a DOMString representing the blob and point the link element towards it
        let url = window.URL.createObjectURL(blob);
        a.href = url;
        a.download = 'output.csv';
        //programatically click the link to trigger the download
        a.click();
        //release the reference to the file by revoking the Object URL
        window.URL.revokeObjectURL(url);           
        }
        el('analyze-button').innerHTML = 'Analyze';
    }

    //xhr.onload = function(e) {
    //    if (this.readyState === 4) {
    //        download('output.csv', data);
            //var response = JSON.parse(e.target.responseText);
            //el('result-label').innerHTML = `Result = ${response['result']}`;
    //    }
    //    el('analyze-button').innerHTML = 'Analyze';
    //}

    // This is new to try download the csv
    //var download = function(filename, content) {
    //    var blob = new Blob([content]);
    //    var evnt =  new Event('click');
    //    $("<a>", {
    //      download: filename,
    //      href: webkitURL.createObjectURL(blob)
    //    }).get(0).dispatchEvent(evnt);
    //};

    var fileData = new FormData();
    fileData.append('upload_file', uploadFiles[0]);
    xhr.send(fileData);
}

