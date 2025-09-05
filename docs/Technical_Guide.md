# Technical Guide

## System Requirements

| Items               | Required  | Recommended |
|---------------------|-----------|-------------|
| **CPU Physical Cores** | 8         | 16           |
| **CPU Speed**       | 2.5GHz     | 3.0GHz      |
| **RAM**             | 24GB        | 32GB        |
| **Storage**         | 30GB       | 35GB        |
| **Operating System** | Ubuntu 20.04 | Ubuntu 24.04 |
| **Bandwidth**       | 30 Mbps/20 Mbps | -        |

## Step-by-Step Guide to Running a Miner/Validator on Ta𝜏su Document Understanding Subnet


### Prerequisites:

- Ensure you have Python installed (preferably version 3.11).
- Install Git for cloning the repository.
- Ensure you have a suitable text editor or IDE for editing configuration files.

### Steps:

1. **Clone the Repository:**

    ```bash
   git clone https://github.com/TatsuProject/Document_Understanding_Subnet.git
   ```


    Navigate to the project directory:

    ```bash
    cd Document_Understanding_Subnet
    ```

2. **Install Dependencies:** 

    Ensure you have pip installed, then install the necessary Python packages:

    ```bash
    pip install -e .
    pip install -r requirements.txt
    ```

3. **Install Tesseract (for miners only):**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

4. **Install the YOLO Checkbox Service (for miners only):**  
   Follow the steps in the link below to install the service:
   
   [Yolo Checkbox Service](https://github.com/TatsuProject/yolo_checkbox_detector)
   
   After installation, ensure the service is running on the same machine as the miner.

6. **Generate Wallet Keys:**

    To participate as a miner, you need to set up wallet keys. You can specify the name you want for the wallet:

    ```bash
    btcli wallet new_coldkey --wallet.name "NAME"
    ```

    This will generate a new wallet and return the coldkey for that wallet. Remember to keep this key secure. Now create a hotkey for the wallet you just created:

    ```bash
    btcli wallet new_hotkey --wallet.name "NAME" --wallet.hotkey default
    ```

    This command will generate a hotkey and coldkey for your miner wallet. Securely store the generated keys. And use any faucet to get Tao in your wallet.
7. **Register Keys**

    This step registers your subnet validator or subnet miner keys to the subnet. Depending on what you are planning to do, you can choose either of these four commands

    Register your miner key to the subnet on **TESTNET**:

    ```bash
    btcli subnet register --netuid 236 --subtensor.network test --wallet.name miner --wallet.hotkey default
    ```


    Register your miner key to the subnet on **MAINNET**:

   ## OR

    ```bash
    btcli subnet register --netuid 54 --subtensor.network finney --wallet.name miner --wallet.hotkey default
    ```


    ## OR
    Register your validator key to the subnet on **TESTNET**:

    ```bash
    btcli subnet register --netuid 236 --subtensor.network test --wallet.name validator --wallet.hotkey default
    ```

    Follow the prompts:

    ## OR
   
    Register your validator key to the subnet on **MAINNET**:

    ```bash
    btcli subnet register --netuid 54 --subtensor.network finney --wallet.name validator --wallet.hotkey default
    ```



    ## Note
    ### Wallets
    If you have your wallets in a folder other than ~./bittensor/wallets, you can give path to that folder in the script
    ```bash
    Document_Understanding_Subnet/template/base/neuron.py
    ```
    in line
    ```bash
    #self.config.wallet.path = "" # set your wallet path here 
    ```


    Use the following command to start the miner on **TESTNET**:

    ```bash
    python3 neurons/miner.py --netuid 236 --subtensor.network test --wallet.name "NAME" --wallet.hotkey default --logging.debug
    ```


    ## OR

    Use the following command to start the Validator on **TESTNET**:

    ```bash
    python3 neurons/validator.py --netuid 236 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug
    ```
    ## Note
    ### Missing Modules Error
    If the following error appears while running miner or vaidator
    ```bash
    ImportError: No module named template
    ```
    you can uncomment the follwoing line in /neurons/miner.py and /neurons/validator.py

    ```bash
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    ```
    and try again.
   
    ## OR

   Use the following command to start the miner on **MAINNET**:

    ```bash
    python neurons/miner.py --netuid --subtensor.network finney --wallet.name "NAME" --wallet.hotkey default --logging.debug
    ```

    ## OR

    Use the following command to start the Validator on **MAINNET**:

    ```bash
    python neurons/validator.py --netuid --subtensor.network finney --wallet.name validator --wallet.hotkey default --logging.debug
    ```


10. **Monitor and Verify:**

    - Monitor the console output to ensure the miner/validator is running correctly.
    - The logging information should help you verify that the miner is correctly processing data and communicating with the network.
