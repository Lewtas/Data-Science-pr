from py7zr import unpack_7zarchive
#shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
# If don'n unpack archive, use first line


def unarchive_test():
    shutil.unpack_archive('input/cifar-10/test.7z')


def unarchive_train():
    shutil.unpack_archive('input/cifar-10/train.7z')
