

## Installation and local usage

- If you are using this app locally, then use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.
```bash
pip install -r requirements
```
- The to run the app just open a terminal and run
```bash
python3 app/main.py serve
```
- Connect to your local host: [http://0.0.0.0:8080](http://0.0.0.0:8080), an interface will appear. 
- Upload a json file with the url containing the test images. You can modify the file **examplePayload.json**
- Click on analyze and wait for the download window.
- The CSV downloaded will contain the images names and labels.

> NOTE: You can also use POSTMAN for the payload. Just make sure to put the KEY: upload_file when you input the json file, and the METHOD: POST. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
