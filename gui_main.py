import sys
import time
import locale
from dialog import Dialog

import numpy as np
import matplotlib.pylab as plt


locale.setlocale(locale.LC_ALL, '')

d = Dialog(dialog="dialog", autowidgetsize=True)

d.set_background_title("Imputlorer")

button_names = {d.OK: "OK",
                d.CANCEL: "Cancel",
                d.HELP: "Help",
                d.EXTRA: "Extra"}

code, path = d.fselect("./data/")
if code == d.ESC:
    d.msgbox("You are exiting Imputlorer :(")
else:
    d.msgbox("The file you have selected is {}".format(path))

X = np.load(path)
plt.plot(X)
plt.show()
# code, tag = d.menu("Some text will show up here",
#                    choices=[("Tag 1", "Item text 1"),
#                             ("Tag 2", "Item text 2"),
#                             ("Tag 3", "Item text 3")
#                             ]
#                    )

# if code == d.ESC:
#     d.msgbox("You exiting Imputlorer!")
# else:
#     text = "You got out of the menu by pressing the {} button".format(button_names[code])

#     if code != d.CANCEL:
#         text += ", and the highlighted entry at that time had tag {!r}".format(tag)

#     d.msgbox(text + ".", width=40, height=10)

d.infobox("Bye Bye ...")
time.sleep(1)
d.clear()
sys.exit(0)
