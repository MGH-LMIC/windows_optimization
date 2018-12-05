#!/usr/bin/env bash

# download ich model
# inceptionv3_ich_sigmoid_U255_nch2
wget -P models/inceptionv3_ich_sigmoid_U255_nch2 https://www.dropbox.com/s/q0alvz3q2igl7ky/model.json
wget -P models/inceptionv3_ich_sigmoid_U255_nch2 https://www.dropbox.com/s/6nwceeuzv3li3zc/weights.h5

# download stone model
# inceptionv3_stone_sigmoid_U255_nch2
wget -P models/inceptionv3_stone_sigmoid_U255_nch2 https://www.dropbox.com/s/sh4u609trmc9upd/model.json
wget -P models/inceptionv3_stone_sigmoid_U255_nch2 https://www.dropbox.com/s/aooc92dz2zof5t1/weights.h5