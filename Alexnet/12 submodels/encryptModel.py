import os, random, struct, binascii
from Crypto.Cipher import AES
from rsa import encrypt
import time

original_size = ((os.path.getsize("submodel_1.pt")) + (os.path.getsize("submodel_2.pt")) 
+ (os.path.getsize("submodel_3.pt")) + (os.path.getsize("submodel_4.pt")) 
+ (os.path.getsize("submodel_5.pt")) + (os.path.getsize("submodel_6.pt")) 
+ (os.path.getsize("submodel_7.pt")) + (os.path.getsize("Main_Submodel2.pt"))) / 1e+6


def encrypt_file(key, in_filename, out_filename, chunksize=64*1024):
    """ Encrypts a file using AES (CBC mode) with the
        given key.

        key:
            The encryption key - a string that must be
            either 16, 24 or 32 bytes long. Longer keys
            are more secure.

        in_filename:
            Name of the input file

        out_filename:
            If None, '<in_filename>.enc' will be used.

        chunksize:
            Sets the size of the chunk which the function
            uses to read and encrypt the file. Larger chunk
            sizes can be faster for some files and machines.
            chunksize must be divisible by 16.
    """

    # iv = ''.join(chr(random.randint(0, 0xFF)) for i in range(16))
    # Initialisation Vector - recommended to be randomnized for each execution of encryption
    # adds 'salt' to the payload
    iv = os.urandom(16)
    encryptor = AES.new(key.encode("utf8"), AES.MODE_CBC, iv)
    filesize = os.path.getsize(in_filename)

    with open(in_filename, 'rb') as infile:
        with open(out_filename, 'wb') as outfile:
            outfile.write(struct.pack('<Q', filesize))
            outfile.write(iv)

            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                elif len(chunk) % 16 != 0:
                    chunk += ' '.encode("utf8") * (16 - len(chunk) % 16)

                outfile.write(encryptor.encrypt(chunk))

def getPass(path):
    with open(path) as f:
        aesSecret = f.readlines()
        pwd = aesSecret[0]
        f.close()

    return pwd

start_time = time.time()

encrypt_file(getPass('secrets/secret_1.txt'), 'submodel_1.pt', 'submodel_1_enc.pt')
encrypt_file(getPass('secrets/secret_2.txt'), 'submodel_2.pt', 'submodel_2_enc.pt')
encrypt_file(getPass('secrets/secret_3.txt'), 'submodel_3.pt', 'submodel_3_enc.pt')
encrypt_file(getPass('secrets/secret_4.txt'), 'submodel_4.pt', 'submodel_4_enc.pt')
encrypt_file(getPass('secrets/secret_5.txt'), 'submodel_5.pt', 'submodel_5_enc.pt')
encrypt_file(getPass('secrets/secret_6.txt'), 'submodel_6.pt', 'submodel_6_enc.pt')
encrypt_file(getPass('secrets/secret_7.txt'), 'submodel_7.pt', 'submodel_7_enc.pt')
encrypt_file(getPass('secrets/secret_8.txt'), 'submodel_8.pt', 'submodel_8_enc.pt')
encrypt_file(getPass('secrets/secret_9.txt'), 'submodel_9.pt', 'submodel_9_enc.pt')
encrypt_file(getPass('secrets/secret_10.txt'), 'submodel_10.pt', 'submodel_10_enc.pt')
encrypt_file(getPass('secrets/secret_11.txt'), 'submodel_11.pt', 'submodel_11_enc.pt')
encrypt_file(getPass('secrets/secret_12.txt'), 'Main_Submodel2.pt', 'Main_Submodel2_enc.pt')

encryption_time = time.time()

encrypted_size = ((os.path.getsize("submodel_1_enc.pt")) + (os.path.getsize("submodel_2_enc.pt")) 
+ (os.path.getsize("submodel_3_enc.pt")) + (os.path.getsize("submodel_4_enc.pt")) 
+ (os.path.getsize("submodel_5_enc.pt")) + (os.path.getsize("submodel_6_enc.pt")) 
+ (os.path.getsize("submodel_7_enc.pt")) + (os.path.getsize("Main_Submodel2_enc.pt"))) / 1e+6

encryption = encryption_time - start_time

print("--------Model has been encrypted--------\n")

print("Encryption Time: {} seconds".format(encryption))
print("Original Size: {} MB \nEncrypted Size: {} MB".format(original_size, encrypted_size))



