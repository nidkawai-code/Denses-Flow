import { df } from './df.js';

const dataObject = [
{ input: [0, 0], output: [0] },
{ input: [0, 1], output: [1] },
{ input: [1, 0], output: [1] },
{ input: [1, 1], output: [0] }
];

const dataset = df.objectData(dataObject);

console.log(JSON.stringify(dataset, null, 2));

const error = [0.1, 0.01, 1.1, 2.2, 0];
const input = [1, 2, 0, -2, -1];

console.log('Linear F' + JSON.stringify(df.linear.forward([...input])));
console.log('Linear B' + JSON.stringify(df.linear.backward([...error])));

const reluOutput = df.relu.forward([...input]);
console.log('Relu F' + JSON.stringify(reluOutput));
console.log('Relu B' + JSON.stringify(df.relu.backward([...error], reluOutput)));

const sigmoidOutput = df.sigmoid.forward([...input]);
console.log('Sigmoid F' + JSON.stringify(sigmoidOutput));
console.log('Sigmoid B' + JSON.stringify(df.sigmoid.backward([...error], sigmoidOutput)));

const tanhOutput = df.tanh.forward([...input]);
console.log('Tanh F' + JSON.stringify(tanhOutput));
console.log('Tanh B' + JSON.stringify(df.tanh.backward([...error], tanhOutput)));

const softmaxOutput = df.softmax.forward([...input]);
console.log('Softmax F' + JSON.stringify(softmaxOutput));
console.log('Softmax B' + JSON.stringify(df.softmax.backward([...error], softmaxOutput)));

console.log('Normal', new Array(3).fill().map(df.normal));

console.log('HEN', new Array(3).fill().map(() => df.HEN(100000)));
console.log('HEU', new Array(3).fill().map(() => df.HEU(100000)));

console.log('Xaviern', new Array(3).fill().map(() => df.xaviern(100000, 100000)));
console.log('Xavieru', new Array(3).fill().map(() => df.xavieru(100000, 100000)));

const model = df.denses({
    layers: [2, 4, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const model2 = df.denses({
    arsitect: [2, 4, 1],
    activates: [df.linear, df.relu, df.sigmoid],
    randoms: [null, df.HEN, df.xavieru]
});

console.log('Model1' + JSON.stringify(model, null, 2));
console.log('Model2' + JSON.stringify(model2, null, 2));

const modelForWalk = df.denses({
    arsitect: [2, 4, 1],
    activates: [df.linear, df.relu, df.sigmoid],
    randoms: [null, df.HEN, df.xavieru]
});

df.walk(modelForWalk, info => {
    console.log('Walk-' + (info.i + 1), JSON.stringify(info));
});

const modelForPredict = df.denses({
    arsitect: [2, 4, 1],
    activates: [df.linear, df.relu, df.sigmoid],
    randoms: [null, df.HEN, df.xavieru]
});

console.log('Predict', JSON.stringify(df.predict([0, 1], modelForPredict)));
console.log('Forward', JSON.stringify(df.forward([0, 1], modelForPredict), null, 2));

const modelForBox = df.denses({
    layers: [2, 4, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const dataObjectXOR = [
{ input: [0, 0], output: [0] },
{ input: [0, 1], output: [1] },
{ input: [1, 0], output: [1] },
{ input: [1, 1], output: [0] }
];

const datasetXor = df.objectData(dataObjectXOR);
let gradientsEnd, errorInput;
for(let epc = 0; epc < 5000; epc++) {
    let loss = 0;
    for(let data = 0; data < datasetXor.x.length; data++) {
        const box = df.forward(datasetXor.x[data], modelForBox);
        const error = [2 * (box[box.length - 1][0] - datasetXor.y[data][0])];
        loss += error[0] * error[0];
        const { gradients, error_input: errorI } = df.backward(error, box, modelForBox);
        gradientsEnd = gradients;
        errorInput = errorI;
        for(let i = 0; i < gradients.length; i++) {
            modelForBox.params[i] -= gradients[i] * 0.1;
        }
    }
    if(epc % 500 === 0) console.log('Epoch: $' + epc + '. | Loss: ~' +loss / datasetXor.x.length);
}

datasetXor.x.forEach(input => {
    console.log('Input: [' + input + ']. Prediksi: [' + df.predict(input, modelForBox) + '.');
});

console.log('Grdients End:' + JSON.stringify(gradientsEnd));
console.log('Error Input Sample End:' + JSON.stringify(errorInput));
console.log('Bobot/bias Flat:' + JSON.stringify(modelForBox.params));

const modelForLosses = df.denses({
    layers: [2, 4, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const dataXORForLosses = [
{ input: [0, 0], output: [0] },
{ input: [0, 1], output: [1] },
{ input: [1, 0], output: [1] },
{ input: [1, 1], output: [0] }
];

const datasetXorForLosses = df.objectData(dataXORForLosses);
for(let epc = 0; epc < 5000; epc++) {
    let loss = [0, 0];
    for(let data = 0; data < datasetXorForLosses.x.length; data++) {
        const box = df.forward(datasetXorForLosses.x[data], modelForLosses);
        const error = df.LIN(box[box.length - 1], datasetXorForLosses.y[data]);
        loss[0] += error.loss[0];
        loss[1] += error.loss[1];
        const { gradients, errorInput: errorI } = df.backward(error.error, box, modelForLosses);
        for(let i = 0; i < gradients.length; i++) {
            modelForLosses.params[i] -= gradients[i] * 0.1;
        }
    }
    if(epc % 500 === 0) console.log('LIN -> Epoch: $' + epc + '. | Loss: ~' + loss[0] / datasetXorForLosses.x.length + ' : ' + loss[1] / datasetXorForLosses.x.length);
}

datasetXorForLosses.x.forEach(input => {
    console.log('LIN predict: [' + input + ']. Prediksi: [' + df.predict(input, modelForLosses) + '].');
});

const modelMSE = df.denses({
    layers: [2, 8, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const xorMSE = [
{ input: [0, 0], output: [0] },
{ input: [0, 1], output: [1] },
{ input: [1, 0], output: [1] },
{ input: [1, 1], output: [0] }
];

const datasetMSE = df.objectData(xorMSE);
for(let epc = 0; epc < 5000; epc++) {
    let loss = 0;
    for(let data = 0; data < datasetMSE.x.length; data++) {
        const box = df.forward(datasetMSE.x[data], modelMSE);
        const error = df.MSE(box[box.length - 1], datasetMSE.y[data]);
        loss += error.loss;
        const { gradients, errorInput: errorI } = df.backward(error.error, box, modelMSE);
        for(let i = 0; i < gradients.length; i++) {
            modelMSE.params[i] -= gradients[i] * 0.1;
        }
    }
    if(epc % 500 === 0) console.log('MSE -> Epoch: $' + epc + '. | Loss: ~' + loss / datasetMSE.x.length);
}

datasetMSE.x.forEach(input => {
    console.log('MSE predict: [' + input + ']. Prediksi: [' + df.predict(input, modelMSE) + '].');
});

const modelMAE = df.denses({
    layers: [2, 8, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const xorMAE = [
{ input: [0, 0], output: [0] },
{ input: [0, 1], output: [1] },
{ input: [1, 0], output: [1] },
{ input: [1, 1], output: [0] }
];

const datasetMAE = df.objectData(xorMAE);
for(let epc = 0; epc < 5000; epc++) {
    let loss = 0;
    for(let data = 0; data < datasetMAE.x.length; data++) {
        const box = df.forward(datasetMAE.x[data], modelMAE);
        const error = df.MAE(box[box.length - 1], datasetMAE.y[data]);
        loss += error.loss;
        const { gradients, errorInput: errorI } = df.backward(error.error, box, modelMAE);
        for(let i = 0; i < gradients.length; i++) {
            modelMAE.params[i] -= gradients[i] * 0.1;
        }
    }
    if(epc % 500 === 0) console.log('MAE -> Epoch: $' + epc + '. | Loss: ~' + loss / datasetMAE.x.length);
}

datasetMAE.x.forEach(input => {
    console.log('MAE predict: [' + input + ']. Prediksi: [' + df.predict(input, modelMAE) + '].');
});

const modelBCE = df.denses({
    layers: [2, 8, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const xorBCE = [
{ input: [0, 0], output: [0] },
{ input: [0, 1], output: [1] },
{ input: [1, 0], output: [1] },
{ input: [1, 1], output: [0] }
];

const datasetBCE = df.objectData(xorBCE);
for(let epc = 0; epc < 5000; epc++) {
    let loss = 0;
    for(let data = 0; data < datasetBCE.x.length; data++) {
        const box = df.forward(datasetBCE.x[data], modelBCE);
        const error = df.BCE(box[box.length - 1], datasetBCE.y[data]);
        loss += error.loss;
        const { gradients, errorInput: errorI } = df.backward(error.error, box, modelBCE);
        for(let i = 0; i < gradients.length; i++) {
            modelBCE.params[i] -= gradients[i] * 0.1;
        }
    }
    if(epc % 500 === 0) console.log('BCE -> Epoch: $' + epc + '. | Loss: ~' + loss / datasetBCE.x.length);
}

datasetBCE.x.forEach(input => {
    console.log('BCE predict: [' + input + ']. Prediksi: [' + df.predict(input, modelBCE) + '].');
});

const modelCCE = df.denses({
    layers: [2, 8, 2],
    activations: [df.linear, df.relu, df.softmax],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const xorCCE = [
{ input: [0, 0], output: [0, 1] },
{ input: [0, 1], output: [1, 0] },
{ input: [1, 0], output: [1, 0] },
{ input: [1, 1], output: [0, 1] }
];

const datasetCCE = df.objectData(xorCCE);
for(let epc = 0; epc < 5000; epc++) {
    let loss = 0;
    for(let data = 0; data < datasetCCE.x.length; data++) {
        const box = df.forward(datasetCCE.x[data], modelCCE);
        const error = df.CCE(box[box.length - 1], datasetCCE.y[data]);
        loss += error.loss;
        const { gradients, errorInput: errorI } = df.backward(error.error, box, modelCCE);
        for(let i = 0; i < gradients.length; i++) {
            modelCCE.params[i] -= gradients[i] * 0.1;
        }
    }
    if(epc % 500 === 0) console.log('CCE -> Epoch: $' + epc + '. | Loss: ~' + loss / datasetCCE.x.length);
}

datasetCCE.x.forEach(input => {
    console.log('CCE predict: [' + input + ']. Prediksi: [' + df.predict(input, modelCCE) + '].');
});

const modelReg = df.denses({
    layers: [2, 4, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const boxReg = df.forward([0, 1], modelReg);
const errorReg = df.MSE(boxReg[boxReg.length - 1], [0]);
const { gradients: gradReg } = df.backward(errorReg.error, boxReg, modelReg);

const gradL12 = [...gradReg];

df.L12(gradL12, modelReg);

console.log('Weights: [' + modelReg.params + ']');
console.log('Grad asli: [' + gradReg + ']');
console.log('GradL12: [' + gradL12 + ']');

const modelSGD = df.denses({
    layers: [2, 8, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const xorSGD = [
{ input: [0, 0], output: [0] },
{ input: [0, 1], output: [1] },
{ input: [1, 0], output: [1] },
{ input: [1, 1], output: [0] }
];

const datasetSGD = df.objectData(xorSGD);
const SGD_states = df.SGD_states(modelSGD);

for(let epc = 0; epc < 5000; epc++) {
    let loss = 0;
    for(let data = 0; data < datasetSGD.x.length; data++) {
        const box = df.forward(datasetSGD.x[data], modelSGD);
        const error = df.BCE(box[box.length - 1], datasetSGD.y[data]);
        loss += error.loss;
        const { gradients, errorInput: errorI } = df.backward(error.error, box, modelSGD);
        
        df.L12(gradients, modelSGD);
        df.SGD(gradients, modelSGD, SGD_states);
    }
    if(epc % 500 === 0) console.log('SGD -> Epoch: $' + epc + '. | Loss: ~' + loss / datasetSGD.x.length);
}

datasetSGD.x.forEach(input => {
    console.log('SGD predict: [' + input + ']. Prediksi: [' + df.predict(input, modelSGD) + '].');
});

const modelMOM = df.denses({
    layers: [2, 8, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const xorMOM = [
{ input: [0, 0], output: [0] },
{ input: [0, 1], output: [1] },
{ input: [1, 0], output: [1] },
{ input: [1, 1], output: [0] }
];

const datasetMOM = df.objectData(xorMOM);
const MOM_states = df.MOM_states(modelMOM);

for(let epc = 0; epc < 5000; epc++) {
    let loss = 0;
    for(let data = 0; data < datasetMOM.x.length; data++) {
        const box = df.forward(datasetMOM.x[data], modelMOM);
        const error = df.BCE(box[box.length - 1], datasetMOM.y[data]);
        loss += error.loss;
        const { gradients, errorInput: errorI } = df.backward(error.error, box, modelMOM);
        
        df.L12(gradients, modelMOM);
        df.MOM(gradients, modelMOM, MOM_states);
    }
    if(epc % 500 === 0) console.log('MOM -> Epoch: $' + epc + '. | Loss: ~' + loss / datasetMOM.x.length);
}

datasetMOM.x.forEach(input => {
    console.log('MOM predict: [' + input + ']. Prediksi: [' + df.predict(input, modelMOM) + '].');
});


const modelRMSp = df.denses({
    layers: [2, 8, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const xorRMSp = [
{ input: [0, 0], output: [0] },
{ input: [0, 1], output: [1] },
{ input: [1, 0], output: [1] },
{ input: [1, 1], output: [0] }
];

const datasetRMSp = df.objectData(xorRMSp);
const RMSp_states = df.RMSp_states(modelRMSp);

for(let epc = 0; epc < 5000; epc++) {
    let loss = 0;
    for(let data = 0; data < datasetRMSp.x.length; data++) {
        const box = df.forward(datasetRMSp.x[data], modelRMSp);
        const error = df.BCE(box[box.length - 1], datasetRMSp.y[data]);
        loss += error.loss;
        const { gradients, errorInput: errorI } = df.backward(error.error, box, modelRMSp);
        
        df.L12(gradients, modelRMSp);
        df.RMSp(gradients, modelRMSp, RMSp_states);
    }
    if(epc % 500 === 0) console.log('RMSp -> Epoch: $' + epc + '. | Loss: ~' + loss / datasetRMSp.x.length);
}

datasetRMSp.x.forEach(input => {
    console.log('RMSp predict: [' + input + ']. Prediksi: [' + df.predict(input, modelRMSp) + '].');
});


const modeladam = df.denses({
    layers: [2, 8, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const xoradam = [
{ input: [0, 0], output: [0] },
{ input: [0, 1], output: [1] },
{ input: [1, 0], output: [1] },
{ input: [1, 1], output: [0] }
];

const datasetadam = df.objectData(xoradam);
const adam_states = df.adam_states(modeladam);

for(let epc = 0; epc < 5000; epc++) {
    let loss = 0;
    for(let data = 0; data < datasetadam.x.length; data++) {
        const box = df.forward(datasetadam.x[data], modeladam);
        const error = df.BCE(box[box.length - 1], datasetadam.y[data]);
        loss += error.loss;
        const { gradients, errorInput: errorI } = df.backward(error.error, box, modeladam);
        
        df.L12(gradients, modeladam);
        df.adam(gradients, modeladam, adam_states);
    }
    if(epc % 500 === 0) console.log('adam -> Epoch: $' + epc + '. | Loss: ~' + loss / datasetadam.x.length);
}

datasetadam.x.forEach(input => {
    console.log('adam predict: [' + input + ']. Prediksi: [' + df.predict(input, modeladam) + '].');
});

const modelFinal = df.denses({
    layers: [2, 8, 1],
    activations: [df.linear, df.relu, df.sigmoid],
    initWeights: [null, df.HEN, df.xavieru],
    biases: 1
});

const xorFinal = [
{ input: [0, 0], output: [0] },
{ input: [0, 1], output: [1] },
{ input: [1, 0], output: [1] },
{ input: [1, 1], output: [0] }
];

const datasetFinal = df.objectData(xorFinal);
const statesFinal = df.adam_states(modelFinal);

df.epoch(5000, epc => {
    let lossNow;
    df.Train({
        dataset: datasetFinal,
        model: modelFinal,
        states: statesFinal,
        loss: df.MSE,
        gradients: (grads, model, state, i) => {
            // i tidak diperlukan, akan berguna saat mau buat mini bactch
            df.L12(grads, model); // elastic net
            df.adam(grads, model, state);
        },
        totalloss: loss => {
            if(epc % 500 === 0) console.log('Epoch: ' + epc, 'Loss: ' + loss);
            lossNow = loss;
        },
        error_input: (err, i) => {
            if(epc % 500 === 0) console.log('Error input: [' + datasetFinal.x[i] + '] = [' + err + ']');
        }
    });
    if(lossNow < 0.01) {
        console.log('Target tercapai: ', lossNow);
        return true;
    }
});

datasetFinal.x.forEach(input => {
    console.log('Prediksi input: [' + input + '] = ' + df.predict(input, modelFinal));
});