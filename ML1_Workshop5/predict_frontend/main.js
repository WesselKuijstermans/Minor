function featureNormalize(citysize) {
    trainingSetMean = 8.159799999999999; // value taken from hands-on 3 'univariate linear regression using a keras/tensorflow NN'
    trainingSetStd = 3.849883995922715; // value taken from hands-on 3 'univariate linear regression using a keras/tensorflow NN'
    return (citysize - trainingSetMean) / trainingSetStd;
}

var my_regression_model;
async function predictProfit(citysize) {
    // only load model once to save resources
    if(!my_regression_model) {
        console.log('load pre-trained regression model')
        my_regression_model = await tf.loadLayersModel('http://localhost:8081/model.json'); // load model into tensorflow.js
    }
    console.log('input: ' + citysize);
    citysize = featureNormalize(citysize); // don't forget to feature-normalize!!
    let result = my_regression_model.predict(tf.tensor([parseFloat(citysize)], [1, 1])).arraySync(); // do prediction
    console.log('result: ' + result);
    return result;
}
//predictProfit('20')