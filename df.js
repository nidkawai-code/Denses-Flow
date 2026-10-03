export const df = {};

// ========== TOOLS ===========
df.exp = x => {
    if(x < 0) return 1 / df.exp(-x);
    let hasil = 1;
    let suku = 1;
    for(let i = 1; i <= 8; i++) {
        suku = suku * (x / i);
        hasil += suku;
    }
    return hasil;
};

df.sqrt = x => {
    let predict = 1;
    for(let i = 0; i < 8; i++) {
        predict = (predict + (x / predict)) * 0.5;
    }
    return predict;
};

df.pow = (base, exp) => {
    if (exp > 5000) return 0;
    let result = 1;
    for (let i = 0; i < exp; i++) result *= base;
    return result;
};

df.log = x => {
    let u = x - 1;
    let suku = u;
    let hasil = 0;

    for (let i = 1; i <= 8; i++) {
        hasil += suku / i;
        suku *= -u;
    }

    return hasil;
};

df.sign = x => x === 0 ? 0 : x > 0 ? 1 : -1;

df.cos = x => {
    let hasil = 1, suku = 1;
    for (let i = 1; i <= 5; i++) {
        suku *= -x * x / (2 * i * (2 * i - 1));
        hasil += suku;
    }
    return hasil;
};


// ============================

/*
[ { input: [], output: [] } ]
 */
df.objectData = object => {
    const dataset = { x: [], y: [] };
    for(let i = 0; i < object.length; i++) {
        dataset.x.push(object[i].input);
        dataset.y.push(object[i].output);
    }
    return dataset;
};
// lulus test

df.linear = {
    forward: output => {
        return output;
    },
    backward: error => {
        return error;
    }
};

// lulus test

df.relu = {
    forward: output => {
        for(let i = 0; i < output.length; i++) {
            output[i] = output[i] > 0 ? output[i] : 0;
        }
        return output;
    },
    backward: (error, activation) => {
        for(let i = 0; i < activation.length; i++) {
            error[i] = activation[i] > 0 ? error[i] : 0;
        }
        return error;
    }
};

// lulus test

df.sigmoid = {
    forward: output => {
        for(let i = 0; i < output.length; i++) {
            output[i] = 1 / (1 + df.exp(-output[i]));
        }
        return output;
    },
    backward: (error, activation) => {
        for(let i = 0; i < activation.length; i++) {
            error[i] = error[i] * (activation[i] * (1 - activation[i]));
        }
        return error;
    }
};

// lulus test

df.tanh = {
    forward: output => {
        for(let i = 0; i < output.length; i++) {
            output[i] = 2 * (1 / (1 + df.exp(-output[i]))) - 1;
        }
        return output;
    },
    backward: (error, activation) => {
        for(let i = 0; i < activation.length; i++) {
            error[i] = error[i] * (1 - (activation[i] * activation[i]));
        }
        return error;
    }
};

// lulus test

df.softmax = {
    forward: output => {
        let sum = 0;
        for(let i = 0; i < output.length; i++) {
            output[i] = df.exp(output[i]);
            sum += output[i];
        }
        for(let i = 0; i < output.length; i++) {
            output[i] = output[i] / sum;
        }
        return output;
    },
    backward: (error, activation) => {
        let sum = 0;
        for(let i = 0; i < activation.length; i++) {
            sum += error[i] * activation[i];
        }
        for(let i = 0; i < activation.length; i++) {
            error[i] = (error[i] - sum) * activation[i];
        }
        return error;
    }
};

// lulus test

df.normal = () => {
    return Math.random() * 2 - 1;
};

// lulus test

df.HEN = fanIn => {
    const u = Math.random() || 1e-15;
    const v = Math.random();
    const step1 = df.sqrt(-2.0 * Math.log(u));
    const step2 = df.cos(2.0 * Math.PI * v);
    const step3 = df.sqrt(2 / fanIn);
    return step1 * step2 * step3;
};

// lulus test

df.HEU = fanIn => {
    const limit = df.sqrt(6 / fanIn);
    return (Math.random() * 2 - 1) * limit;
};

// lulus test

df.xaviern = (fanIn, fanOut) => {
    const u = Math.random() || 1e-15;
    const v = Math.random();
    const step1 = df.sqrt(-2.0 * Math.log(u));
    const step2 = df.cos(2.0 * Math.PI * v);
    const step3 = df.sqrt(2 / (fanIn + fanOut));
    return step1 * step2 * step3;
};

// lulus test

df.xavieru = (fanIn, fanOut) => {
    const limit = df.sqrt(6 / (fanIn + fanOut));
    return (Math.random() * 2 - 1) * limit;
};

// lulus test

df.denses = (config = {}) => {
    const bias = config.biases || 0;
    const layers = config.layers || config.arsitect || [];
    const activations = config.activations || config.activates;
    const randoms = config.initWeights || config.randoms;
    const model = {
        layers,
        activations,
        params: []
    };
    const params = model.params;
    for(let layer = 0; layer < layers.length - 1; layer++) {
        const fanIn = layers[layer];
        const fanOut = layers[layer + 1];
        for(let unit = 0; unit < fanOut; unit++) {
            for(let weight = 0; weight < fanIn; weight++) {
                params.push(randoms[layer + 1](fanIn, fanOut));
            }
            params.push(bias);
        }
    }
    return model;
};

// lulus test

df.walk = (model, fn) => {
    const { layers, params } = model;
    let pointer = 0;
    for(let layer = 0; layer < layers.length - 1; layer++) {
        const fanIn = layers[layer];
        const fanOut = layers[layer + 1];
        for(let unit = 0; unit < fanOut; unit++) {
            for(let weight = 0; weight < fanIn; weight++) {
                fn({
                    type: 'weight',
                    i: pointer,
                    val: params[pointer++],
                    l: layer,
                    u: unit,
                    w: weight
                });
            }
            fn({
                type: 'bias',
                i: pointer,
                val: params[pointer++],
                l: layer,
                u: unit,
                w: false
            });
        }
    }
};

// lulus test

df.predict = (input, model) => {
    const { layers, activations, params } = model;
    input = activations[0].forward(input);
    let pointer = 0;
    for(let layer = 0; layer < layers.length - 1; layer++) {
        const fanIn = layers[layer];
        const fanOut = layers[layer + 1];
        const output = [];
        for(let unit = 0; unit < fanOut; unit++) {
            let sum = 0;
            for(let weight = 0; weight < fanIn; weight++) {
                sum += input[weight] * params[pointer++];
            }
            output.push(sum + params[pointer++]);
        }
        input = activations[layer + 1].forward(output);
    }
    return input;
};

// lulus test

df.forward = (input, model) => {
    const { layers, activations, params } = model;
    input = activations[0].forward(input);
    const box = [input];
    let pointer = 0;
    for(let layer = 0; layer < layers.length - 1; layer++) {
        const fanIn = layers[layer];
        const fanOut = layers[layer + 1];
        const output = [];
        for(let unit = 0; unit < fanOut; unit++) {
            let sum = 0;
            for(let weight = 0; weight < fanIn; weight++) {
                sum += input[weight] * params[pointer++];
            }
            output.push(sum + params[pointer++]);
        }
        const activation = activations[layer + 1].forward(output);
        box.push(activation);
        input = activation;
    }
    return box;
};

// lulus test

df.backward = (error, box, model) => {
    const { layers, activations, params } = model;
    const gradients = [];
    let pointer = params.length - 1;
    for(let layer = layers.length - 1; layer > 0; layer--) {
        const fanIn = layers[layer - 1];
        const fanOut = layers[layer];
        const errorInput = box[layer - 1].map(() => 0);
        const dactivation = activations[layer].backward(error, box[layer]);
        for(let unit = fanOut - 1; unit > -1; unit--) {
            const delta = dactivation[unit];
            gradients[pointer--] = delta;
            for(let weight = fanIn - 1; weight > -1; weight--) {
                gradients[pointer] = delta * box[layer - 1][weight];
                errorInput[weight] = delta * params[pointer--]; 
            }
        }
        error = errorInput;
    }
    return { gradients, error_input: activations[0].backward(error, box[0]) };
};

// lulus test

df.LIN = (prediksi, target) => {
    const loss = [0, 0];
    const error = new Array(prediksi.length);
    const N = prediksi.length;
    for(let i = 0; i < prediksi.length; i++) {
        const err = prediksi[i] - target[i];
        err > 0 ? loss[1] += err : loss[0] += err;
        error[i] = err / N;
    }
    return { loss: [loss[0] / N, loss[1] / N], error };
};

// lulus test

df.MSE = (prediksi, target) => {
    let loss = 0;
    const error = new Array(prediksi.length);
    const N = prediksi.length;
    for(let i = 0; i < prediksi.length; i++) {
        const err = prediksi[i] - target[i];
        loss += err * err;
        error[i] = (2 * err) / N;
    }
    return { loss: loss / N, error };
};

// lulus test

df.MAE = (prediksi, target) => {
    let loss = 0;
    const error = new Array(prediksi.length);
    const N = prediksi.length;
    for(let i = 0; i < prediksi.length; i++) {
        const err = prediksi[i] - target[i];
        loss += err >= 0 ? err : -err;
        error[i] = (err === 0 ? 0 : err > 0 ? 1 : -1) / N;
    }
    return { loss: loss / N, error };
};

// lulus test

df.BCE = (prediksi, target) => {
    let loss = 0;
    const error = new Array(prediksi.length);
    const N = prediksi.length;
    for(let i = 0; i < prediksi.length; i++) {
        const p = prediksi[i] + 1e-15;
        const t = target[i];
        loss += -(t * df.log(p) + (1 - t) * df.log(1 - p));
        error[i] = ((p - t) / (p * (1 - p))) / N;
    }
    return { loss: loss / N, error };
};

// lulus test

df.CCE = (prediksi, target) => {
    let loss = 0;
    const error = new Array(prediksi.length);
    const N = prediksi.length;
    for(let i = 0; i < prediksi.length; i++) {
        const p = prediksi[i] + 1e-15;
        const t = target[i];
        const eps = 1e-15;
        loss += -t * df.log(p);
        error[i] = (-t / p) / N;
    }
    return { loss: loss / N, error };
};

// lulus test

df.L12 = (gradients, model, config = {}) => {
    const { lambda = 0.001, ratio = 0.5 } = config;
    df.walk(model, info => {
        if(info.type === 'weight') {
            const l1 = df.sign(info.val) * ratio;
            const l2 = (1 - ratio) * info.val;
            gradients[info.i] += lambda * (l1 + l2); 
        }
    });
};

// lulus test

df.SGD_states = (model, config = {}) => {
    const { lr = 0.01 } = config;
    return { lr };
};

df.SGD = (gradients, model, states) => {
    const params = model.params;
    for(let i = 0; i < gradients.length; i++) {
        params[i] -= gradients[i] * states.lr;
    }
};

// lulus test

df.MOM_states = (model, config = {}) => {
    const { lr = 0.01, beta = 0.9 } = config;
    return { lr, beta, vel: new Array(model.params.length).fill(0) };
};

df.MOM = (gradients, model, states) => {
    const { lr, beta, vel } = states;
    const params = model.params;
    for(let i = 0; i < gradients.length; i++) {
        vel[i] = (vel[i] * beta) + (gradients[i] * lr);
        params[i] -= vel[i];
    }
};

// lulus test

df.RMSp_states = (model, config = {}) => {
    const { lr = 0.001, beta = 0.99, eps = 1e-8 } = config;
    return { lr, beta, eps, squ: new Array(model.params.length).fill(0) };
};


df.RMSp = (gradients, model, states) => {
    const { lr, beta, squ, eps } = states;
    const params = model.params;
    for(let i = 0; i < gradients.length; i++) {
        squ[i] = squ[i] * beta + (1 - beta) * (gradients[i] * gradients[i]);
        params[i] -= (lr / (df.sqrt(squ[i]) + eps)) * gradients[i];
    }
};

// lulus test

df.adam_states = (model, config = {}) => {
    const {
        lr = 0.001,
        beta1 = 0.9,
        beta2 = 0.999,
        eps = 1e-8
    } = config;
    return {
        lr, beta1, beta2, eps,
        mom: new Array(model.params.length).fill(0),
        vel: new Array(model.params.length).fill(0),
        step: 0
    }
};

df.adam = (gradients, model, states) => {
    const { lr, beta1, beta2, eps, mom, vel } = states;
    const params = model.params;
    states.step++;
    const step = states.step;
    const bc1 = 1 - df.pow(beta1, step);
    const bc2 = 1 - df.pow(beta2, step);
    const LRT = lr * (df.sqrt(bc2) / bc1);
    for(let i = 0; i < gradients.length; i++) {
        mom[i] = (mom[i] * beta1) + (1 - beta1) * gradients[i];
        vel[i] = (vel[i] * beta2) + (1 - beta2) * (gradients[i] * gradients[i]);
        params[i] -= (LRT * mom[i]) / (df.sqrt(vel[i]) + eps);
    }
};

// lulus test

df.epoch = (val, fn) => {
    for(let i = 0; i < val; i++) {
        if(fn(i) === true) break;
    }
};

// lulus test

df.Train = config => {
    let sumloss = 0;
    const {
        dataset,
        model,
        states,
        loss,
        gradients,
        totalloss = () => {},
        error_input = () => {}
    } = config;
    const { x, y } = dataset;
    for(let i = 0; i < x.length; i++) {
        const input = x[i];
        const target = y[i];
        const box = df.forward(input, model);
        const prediksi = box[box.length - 1];
        const { loss: lossx, error } = loss(prediksi, target);
        if(Array.isArray(lossx)) {
            if(!Array.isArray(sumloss)) sumloss = [0, 0];
            sumloss[0] += lossx[0];
            sumloss[1] += lossx[1];
        } else {
            sumloss += lossx;
        }
        const { gradients: grads, error_input: errin } = df.backward(error, box, model);
        gradients(grads, model, states, i);
        error_input(errin, i);
    }
    totalloss(!Array.isArray(sumloss) ? sumloss / x.length : [sumloss[0] / x.length, sumloss[1] / x.length]);
};

// lulus test
