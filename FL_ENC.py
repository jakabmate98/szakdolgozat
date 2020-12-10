from charm.toolbox.PKEnc import PKEnc
from charm.toolbox.ecgroup import G, ECGroup
from charm.toolbox.eccurve import prime192v2
import tensorflow as tf
import numpy as np
import random
import math
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras import backend as K_s
import gc
from datetime import datetime
import shutil
import os

# Server is weighting given ciphertext
def serverWeighting(ct, client_weight):
    if (client_weight == 0):
        return nulla
    if (client_weight == 1):
        return ct
    c1 = ct['c1']
    c2 = ct['c2']
    c1_ = c1
    c2_ = c2
    for i in range(client_weight - 1):
        c1_ = c1_ * c1
        c2_ = c2_ * c2
    return ElGamalCipher({'c1': c1_, 'c2': c2_})


def train_accuracy(agg_enc_model):
    if debug:
        print("Decrypt start:", datetime.now())
    w_t = decrypt_model(agg_enc_model, skala_suly * skala_client)
    if debug:
        print("Decrypt end:", datetime.now())
        print("Encrypt start:", datetime.now())
    w_t_enc = encryptModel(w_t, public_key)
    if debug:
        print("Encrypt end:", datetime.now())
    model = create_model()
    model.set_weights(w_t)
    total = 0
    for i in range(group_size):
        X_train = np.load("client_data/X_train_%d.npy" % i)
        Y_train = np.load("client_data/Y_train_%d.npy" % i)
        total += model.evaluate(X_train, Y_train, batch_size=len(Y_train), verbose=0)[1]

    return total / group_size, w_t_enc


def test_accuracy(agg_enc_model):
    if debug:
        print("Decrypt start:", datetime.now())
    w_t = decrypt_model(agg_enc_model, skala_suly * skala_client)
    if debug:
        print("Decrypt end:", datetime.now())
        print("Encrypt start:", datetime.now())
    w_t_enc = encryptModel(w_t, public_key)
    if debug:
        print("Encrypt end:", datetime.now())
    model = create_model()
    model.set_weights(w_t)

    X_train = np.load("server_data/X_test.npy")
    Y_train = np.load("server_data/Y_test.npy")
    total = model.evaluate(X_train, Y_train, batch_size=len(Y_train), verbose=0)[1]

    return total, w_t_enc


def create_model():
    # not too complicated model (only testing purposes)
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    return model


# Splitting input and output data into train, validation and test datasets
def dataset_split(X, Y, valid_split, test_split):
    v_start = int(len(Y) * (1 - valid_split - test_split))
    t_start = int(len(Y) * (1 - test_split))
    X_train, Y_train = X[:v_start], Y[:v_start]
    X_valid, Y_valid = X[v_start:t_start], Y[v_start:t_start]
    X_test, Y_test = X[t_start:], Y[t_start:]
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


# Set clients' weights (integer)
def set_client_weights(group, client_accuracies, c_w, r_cw):
    sum_cw = 0
    amount_of_say = {}

    # Choose weighting algorithm
    for k in group:
        amount_of_say[k] = 0.5 * math.log(client_accuracies[k] / (1 - client_accuracies[k]))
        # c_w[k] = 1/group_size*math.exp(amount_of_say[k]) #adaboost algorithm (previous performance does not matter)
        c_w[k] = c_w[k] * math.exp(amount_of_say[k])  # adaboost algorithm (previous performance matters)
        sum_cw += c_w[k]

    # normalizing (sum has to be 1)
    for k in group:
        c_w[k] = c_w[k] / sum_cw
        r_cw[k] = int(round(c_w[k] * skala_suly))
    dif = skala_suly - sum(r_cw.values())
    r_cw[max(r_cw, key=r_cw.get)] += dif


# Count differentialities in 2 ordered list
def count_dif_lists(list1, list2):
    dif = 0
    for i in range(min(len(list1), len(list2))):
        if (list1[i] != list2[i]):
            dif += 1
    return dif


class ElGamalCipher(dict):
    def __init__(self, ct):
        if type(ct) != dict: assert False, "Not a dictionary!"
        if not set(ct).issubset(['c1', 'c2']): assert False, "'c1','c2' keys not present."
        dict.__init__(self, ct)

    def __mul__(self, other):
        lhs_c1 = dict.__getitem__(self, 'c1')
        lhs_c2 = dict.__getitem__(self, 'c2')
        rhs_c1 = dict.__getitem__(other, 'c1')
        rhs_c2 = dict.__getitem__(other, 'c2')
        return ElGamalCipher({'c1': lhs_c1 * rhs_c1, 'c2': lhs_c2 * rhs_c2})

    def __div__(self, other):
        lhs_c1 = dict.__getitem__(self, 'c1')
        lhs_c2 = dict.__getitem__(self, 'c2')
        rhs_c1 = dict.__getitem__(other, 'c1')
        rhs_c2 = dict.__getitem__(other, 'c2')
        return ElGamalCipher({'c1': lhs_c1 / rhs_c1, 'c2': lhs_c2 / rhs_c2})


# Server aggregates clients' encrypted models
def agg_models(w_k_t_1, group):
    if debug:
        print("agg start", datetime.now())
    agg_enc_model = []
    agg_enc_model1 = []
    for i in range(len(w_k_t_1[group[0]][0])):
        agg_enc_model2 = []
        for j in range(len(w_k_t_1[group[0]][0][i])):
            agg_enc_model2.append(w_k_t_1[group[0]][0][i][j])
        agg_enc_model1.append(agg_enc_model2)
    agg_enc_model.append(agg_enc_model1)
    agg_enc_model3 = []
    for i in range(len(w_k_t_1[group[0]][1])):
        agg_enc_model3.append(w_k_t_1[group[0]][1][i])
    agg_enc_model.append(agg_enc_model3)

    for i in group[1:]:

        for j in range(len(w_k_t_1[i][0])):
            for k in range(len(w_k_t_1[i][0][j])):
                agg_enc_model[0][j][k] = agg_enc_model[0][j][k] * w_k_t_1[i][0][j][k]
        for j in range(len(w_k_t_1[i][1])):
            agg_enc_model[1][j] = agg_enc_model[1][j] * w_k_t_1[i][1][j]
    return agg_enc_model


# Run server (fed learning)
def run_server(w_t_enc2):
    results = []
    # Client exact weights (float)
    c_w = {}
    # Scaled weights of clients (integer)
    r_cw = {}
    for i in range(client_num):
        c_w[i] = 1 / group_size

    X_val = np.load("server_data/X_test.npy")
    Y_val = np.load("server_data/Y_test.npy")

    for round_num in range(rounds):
        print("Round:", round_num + 1)

        w_k_t_1 = {}
        client_accuracies = {}
        client_predictions = {}
        # Select clients for aggregation
        group = np.random.choice(range(client_num), group_size, replace=False)  # select a set of neighbors
        # Load the global model (initially, it is a random model)

        # Update the params of the global model by each selected client and get the accuracy of the modified model (by each client) on server's data
        for i in group:
            if debug:
                print("clientupdate start", datetime.now())
            client_predictions[i], w_k_t_1[i] = ClientUpdate(i, w_t_enc2, X_val)
            if debug:
                print("clientupdate end", datetime.now())
            # Calculating accuracy based on differentialities of client predictions and Y_val
            dif = count_dif_lists(Y_val, client_predictions[i])
            client_accuracies[i] = (len(Y_val) - dif) / len(Y_val)
            K_s.clear_session()

        # Get weights of clients by their performance
        set_client_weights(group, client_accuracies, c_w, r_cw)

        # Compute the central model's encrypted update
        for i in group:
            if debug:
                print("Weight of", i, r_cw[i])
            for j in range(len(w_k_t_1[i][0])):
                for k in range(len(w_k_t_1[i][0][j])):
                    w_k_t_1[i][0][j][k] = serverWeighting(w_k_t_1[i][0][j][k], r_cw[i])
            for j in range(len(w_k_t_1[i][1])):
                w_k_t_1[i][1][j] = serverWeighting(w_k_t_1[i][1][j], r_cw[i])

        agg_enc_model = agg_models(w_k_t_1, group)

        del w_k_t_1
        del w_t_enc2
        # acc, w_t_enc2 = train_accuracy(agg_enc_model)
        acc, w_t_enc2 = test_accuracy(agg_enc_model)
        # Report train and test accuracies of the common model
        results.append(acc)
        # print("Train accuracy: %.3f\n" % results[len(results)-1])
        print("Test accuracy: %.3f\n" % results[len(results) - 1])
        del agg_enc_model
        K_s.clear_session()
        gc.collect()


# Decrypt aggregated model
def decrypt_model(w_t_enc, scale):
    w_t = []
    w_t1 = []
    for i in range(len(w_t_enc[0])):
        w_t2 = []
        for j in range(len(w_t_enc[0][i])):
            m = el.decryptZp(public_key, secret_key, w_t_enc[0][i][j])
            w_t2.append(search(anotherMapper, m) / scale)
        w_t1.append(np.asarray(w_t2))
    w_t.append(np.asarray(w_t1))
    w_t3 = []
    for i in range(len(w_t_enc[1])):
        m = el.decryptZp(public_key, secret_key, w_t_enc[1][i])
        w_t3.append(search(anotherMapper, m) / scale)

    w_t.append(np.asarray(w_t3))
    return w_t

# Updating a Client in one round
def ClientUpdate(k, w_t_enc, X_val):
    w_t_1 = decrypt_model(w_t_enc, skala_client)
    model = create_model()
    model.set_weights(w_t_1)
    # Load the client's data
    X_train = np.load("client_data/X_train_%d.npy" % k)
    Y_train = np.load("client_data/Y_train_%d.npy" % k)

    # Update the global model trained on the client's data
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    # Save the predictions
    pred = model.predict_classes(X_val)
    # Compute the signs of the client's update
    update = model.get_weights()
    del model
    update_enc = []
    update_enc.append(encryptModel(update, public_key))
    del update

    return pred, update_enc[0]

# Initialization of encryption
def initEncryption():
    groupObj = ECGroup(prime192v2)
    el = ElGamalElliptic(groupObj)
    (public_key, secret_key) = el.keygen()
    # rnd = group.zr(group.encode(b"T1tk0$_sz4m_v4gy0k!!!"))
    rnd = group.zr(group.encode(b" "))
    return el, public_key, secret_key, rnd

# Encryption
class ElGamalElliptic(PKEnc):
    def __init__(self, groupObj, p=0, q=0):
        PKEnc.__init__(self)
        global group
        group = groupObj
        if group.groupSetting() == 'integer':
            group.p, group.q, group.r = p, q, 2

    def keygen(self, secparam=1024):
        if group.groupSetting() == 'integer':
            print("Group setting is not elliptic_curve. I cannot generate elliptic curve based keys.")
            if group.p == 0 or group.q == 0:
                group.paramgen(secparam)
            g = group.randomGen()
        elif group.groupSetting() == 'elliptic_curve':
            g = group.random(G)

        # x is private, g is public param
        x = group.random();
        h = g ** x
        if debug:
            print('Public parameters:')
            print('h => %s' % h)
            print('g => %s' % g)
            print('Secret key:')
            print('x => %s' % x)
        pk = {'g': g, 'h': h}
        sk = {'x': x}
        return pk, sk

    def encryptZp(self, pk, M):
        y = group.random()
        c1 = pk['g'] ** y
        s = pk['h'] ** y
        m = pk['g'] ** M
        c2 = m * s
        return ElGamalCipher({'c1': c1, 'c2': c2})

    def encryptByte(self, pk, M):
        y = group.random()
        c1 = pk['g'] ** y
        s = pk['h'] ** y
        # check M and make sure it's right size
        c2 = group.encode(M) * s
        return ElGamalCipher({'c1': c1, 'c2': c2})

    def decryptZp(self, pk, sk, c):
        s = c['c1'] ** sk['x']
        m1 = c['c2'] * (s ** -1)
        return m1

    def decryptByte(self, pk, sk, c):
        s = c['c1'] ** sk['x']
        m = c['c2'] * (s ** -1)
        if group.groupSetting() == 'integer':
            M = group.decode(m % group.p)
        elif group.groupSetting() == 'elliptic_curve':
            M = group.decode(m)
        if debug: print('m => %s' % m)
        if debug: print('dec M => %s' % M)
        return M

# encrypt client's model
def encryptModel(weights, pk):
    model_ency = []

    model_enc1y = []
    for i in range(len(weights[0])):
        model_enc2y = []
        for j in range(len(weights[0][i])):
            M = mapper[round(weights[0][i][j] * skala_client)]
            model_enc2y.append(el.encryptZp(pk, M))
        model_enc1y.append(model_enc2y)
    model_ency.append(model_enc1y)

    model_enc3y = []
    for i in range(len(weights[1])):
        M = mapper[round(weights[1][i] * skala_client)]
        model_enc3y.append(el.encryptZp(pk, M))
    model_ency.append(model_enc3y)

    return model_ency

# Create databases (for clients and server) based on MNIST database
def generate_clients_database(good_guys):
    shutil.rmtree('client_data', ignore_errors=True)
    shutil.rmtree('server_data', ignore_errors=True)
    shutil.rmtree('out', ignore_errors=True)
    os.mkdir("server_data")
    os.mkdir("client_data")
    os.mkdir("out")

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    training_data = []
    for i in range(len(x_train)):
        training_data.append([x_train[i], y_train[i]])
    random.shuffle(training_data)

    X = []
    y = []

    border = int(good_guys / client_num * 60000)

    # first half get correct labels
    for features, label in training_data[:border]:
        X.append(features)
        y.append(label)

    # other half get incorrect labels
    for features, label in training_data[border:]:
        X.append(features)
        # y.append(label)
        # all of them got random labels
        y.append(random.randint(0, 9))

    index = 0
    for i in range(client_num):
        np.save("client_data/X_train_%d.npy" % i, X[int(index):int(index + len(x_train) / client_num)])
        np.save("client_data/Y_train_%d.npy" % i, y[int(index):int(index + len(x_train) / client_num)])
        index += len(x_train) / client_num

    np.save("server_data/X_test.npy", x_test)
    np.save("server_data/Y_test.npy", y_test)


def searchbyValue(mapper, v):
    for integer, value in mapper.items():
        if value == v:
            return integer


def search(anothermapper, v):
    for i in range(skala_client * skala_suly):
        if anothermapper[i] == v:
            return i
        if anothermapper[-i] == v:
            return -i


def int_to_zp(rnd):
    if debug:
        print("int_to_zp start", datetime.now())

    mapper = {}
    mapper[0] = rnd - rnd
    for i in range(1, skala_client * skala_suly):
        mapper[i] = mapper[i - 1] + rnd
        mapper[-i] = mapper[-i + 1] - rnd

    if debug:
        print("int_to_zp end", datetime.now())
    return mapper


def zp_to_gzp(public_key):
    if debug:
        print("zp_to_gzp start", datetime.now())
    anotherMapper = {}
    anotherMapper[0] = public_key['g'] ** mapper[0]
    multiplierplus = public_key['g'] ** mapper[1]
    multiplierneg = public_key['g'] ** mapper[-1]
    for i in range(1, skala_client * skala_suly):
        anotherMapper[i] = anotherMapper[i - 1] * multiplierplus
        anotherMapper[-i] = anotherMapper[-i + 1] * multiplierneg
    if debug:
        print("zp_to_gzp end", datetime.now())

    return anotherMapper


debug = False
client_num = 100
good_guys = 100     # number of good clients
group_size = 4      # number of clients selected per round
rounds = 5
epochs = 1
batch_size = 32
seed_init = 1
skala_client = 100
skala_suly = 20

# This is where the global model is saved
MODEL_FILE = "out/model.h5"

if debug:
    print("Client number:", client_num)
    print("Good client number:", good_guys)
    print("Group size:", group_size)
    print("Rounds:", rounds)
    print("Epochs per round:", epochs)
    print("Batch size:", batch_size)
    print("Scale of model parameters:", skala_client)
    print("Scale of server weights:", skala_suly)

el, public_key, secret_key, rnd = initEncryption()

mapper = int_to_zp(rnd)
anotherMapper = zp_to_gzp(public_key)
generate_clients_database(good_guys)
nulla = el.encryptZp(public_key, 0)
np.random.seed(seed=seed_init)
init_model_w = create_model().get_weights()

K_s.clear_session()
gc.collect()

run_server(encryptModel(init_model_w, public_key))

from itertools import combinations
import math
from datetime import datetime


def test_accuracy2(modell, X_val, Y_val):
    return modell.evaluate(X_val, Y_val, batch_size=len(Y_val), verbose=0)[1]

# Data shapley (parameters: group, model weights)
def shapley(lst, models):
    print(datetime.now())
    lst.sort()
    combs = []
    aggmodels = {}
    accuracies = {}
    combs.append([])
    for i in range(1, len(lst) + 1):
        els = [list(x) for x in combinations(lst, i)]
        combs.extend(els)
    aggmodels[0] = None
    accuracies[0] = 0.1
    for comb in combs[1:]:
        modell = []
        modell.append(tf.convert_to_tensor(models[comb[0]][0], dtype=tf.float64))
        modell.append(tf.convert_to_tensor(models[comb[0]][1], dtype=tf.float64))
        for i in range(1, len(comb)):
            for j in range(len(models[comb[i]])):
                modell[j] += tf.convert_to_tensor(models[comb[i]][j], dtype=tf.float64)

        for i in range(len(models[comb[0]])):
            modell[i] / len(comb)
        aggmodels[combs.index(comb)] = modell

    X_val = np.load("data/X_test.npy")
    Y_val = np.load("data/Y_test.npy")
    modell = create_model()
    for i in range(1, len(aggmodels)):
        modell.set_weights(aggmodels[i])
        accuracies[i] = test_accuracy2(modell, X_val, Y_val)
        i += 1

    shapley_values = np.zeros(len(lst))

    for i in range(len(lst)):
        shapley_value = 0
        for comb in combs:
            comb_shapley = 0
            if i not in comb:
                comb_i = comb + [i]
                comb_i.sort()
                comb_shapley = accuracies[combs.index(comb_i)] - accuracies[combs.index(comb)]
                mult = math.factorial(len(comb)) * math.factorial(len(lst) - len(comb) - 1) / math.factorial(len(lst))
                comb_shapley *= mult
            shapley_value += comb_shapley
        shapley_values[lst[i]] = shapley_value

    print(shapley_values)
    print(datetime.now())

    return shapley_values

