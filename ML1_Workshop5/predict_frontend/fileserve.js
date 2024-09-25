/*
This node application serves the pre-trained Keras/Tensorflow model, from 
hands-on 3 'univariate linear regression using a keras/tensorflow NN' to a front-end.

To run this application: command prompt> node fileserve.js
To use this application: open browser with http://localhost:8081/<some filename>
*/

const express = require('express');
var cors = require('cors');
const app = express();
const port = 8081;

app.get('/', (req, res) => res.send('Hello World!'));
app.listen(port, () => console.log(`Example app listening at http://localhost:${port}`));
app.use(cors());

app.use(express.static('.'));
app.use(express.static('./univariateKerasRegression_tfjs_model'));
