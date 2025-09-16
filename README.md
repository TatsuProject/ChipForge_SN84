# ChipForge (SN84)

ChipForge (SN84) introduces the first digital design subnet for decentralized hardware innovation. This subnet enables miners to compete in designing real silicon. Processor development is organized into on-chain challenges — spanning AI accelerators, cryptographic modules, mini-GPUs, and other critical components. Participants download specifications, leverage AI tools, and submit complete Verilog/SystemVerilog implementations. The highest-quality designs earn rewards while contributing to fully manufacturable chips.

In the short term, ChipForge focuses on advancing digital hardware design, with future applications across IoT, robotics, edge devices, and post-quantum security. Our roadmap includes progressing from design to full-scale fabrication within a year. Revenue from design IPs and fabricated chips will be reinvested into the ecosystem, ensuring sustainable value creation. Backed by the Tatsu validator team and open to strategic partnerships, ChipForge marks the beginning of a new era — decentralized, collaborative, and on-chain digital design.

## Overview

ChipForge operates as a competitive platform where:
- **Miners** submit hardware design solutions (Verilog/SystemVerilog)
- **Validators** evaluate submissions using industry-standard EDA tools (Verilator, Yosys, Icarus, Openlane)
- **Challenges** are rotated periodically with different design requirements
- **Rewards** are distributed based on performance metrics and competitive scoring - This is a winner-takes-all reward mechanism

## Architecture

### Core Components

1. **Chipforge Challenge Server** - Manages challenges, submissions, and evaluations
2. **Miners** - Submit optimized hardware designs for active challenges
3. **Validators** - Download, evaluate, and score miner submissions against provided test benches and test cases
4. **Chipforge EDA Server** - Performs synthesis, place & route, and timing analysis

### Workflow

```
1. Challenge Activation → 2. Miner Submissions → 3. Batch Creation → 
4. Validator Downloads → 5. EDA Evaluation → 6. Score Submission → 
7. Weight Setting → 8. Challenge Completion
```

## Getting Started

### Prerequisites

- Python 3.12
- Bittensor wallet with registered hotkey
- Access to Chipforge Challenge Server API
- Chipforge EDA Server (for validators)

### For Miners

#### Installation

To create wallets, visit this:
```
https://docs.learnbittensor.org/btcli
```

```bash
# Clone the repository
git clone https://github.com/TatsuProject/ChipForge_SN84
cd chipforge-subnet

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

#### Running a Miner

Set parameters in .env file and run:
```bash
./start_miner
```

#### Running with nohup (background process)

To run miner/validator through nohup (no hang up, an alternative of pm2):
```bash
nohup ./start_miner.sh > miner.log 2>&1 &
```

To show logs in real-time:
```bash
tail -f miner.log
```

To terminate the process:
```bash
# Find the process ID:
ps aux | grep miner

# Kill by PID:
kill <PID>

# Or kill by process name:
pkill -f "miner.py"

# Force kill if needed:
kill -9 <PID>
```

#### Miner Responsibilities

- Download challenge
- Solve it 
- Submit the solution

#### Submitting Solutions

```bash
# set wallet and path to zip file in .env file and run this
./submit_score.sh
```

#### Solution Format

Solutions must be packaged as ZIP files containing:
- Verilog/SystemVerilog source files (.v, .sv)
- Testbench files (optional)
- Constraint files (optional)
- README with design description (optional)

#### Rate Limits
- Challenge server accepts 5 requests per IP per hour.
- One hotkey is allowed to submit a maximum 5 solutions for a specific challenge.

### For Validators

#### Running a Validator

Pull and run Chipforge EDA Server:
```
https://github.com/TatsuProject/chipforge_eda_server
```

Set parameters in .env file and run:
```bash
./start_validator
```

#### Running with nohup (background process)

To run validator through nohup (no hang up, an alternative of pm2):
```bash
nohup ./start_validator.sh > validator.log 2>&1 &
```

To show logs in real-time:
```bash
tail -f validator.log
```

To terminate the process:
```bash
# Find the process ID:
ps aux | grep validator

# Kill by PID:
kill <PID>

# Or kill by process name:
pkill -f "validator.py"

# Force kill if needed:
kill -9 <PID>
```

#### Validator Responsibilities

- Download submissions from active batches
- Evaluate designs using EDA tools
- Submit scores based on multiple metrics:
  - **Functionality** (0-100): Correctness and testbench passing
  - These will be part of validation mechanism in future:
    - **Area** (0-100): Resource utilization efficiency
    - **Delay** (0-100): Timing performance
    - **Power** (0-100): Power consumption optimization
    - **Overall** (0-100): Weighted combination of all metrics

## Evaluation Metrics

### Scoring System

Each submission is evaluated across four key metrics:

1. **Functionality Score** (100% weight)
   - Testbench pass/fail status
   - Functional correctness verification
   - Compliance with specifications

2. **Area Score** (TBD)
   - LUT utilization
   - Register usage
   - Memory block efficiency
   - Overall resource optimization

3. **Delay Score** (TBD)
   - Maximum frequency achieved
   - Critical path timing
   - Setup/hold time margins

4. **Power Score** (TBD)
   - Static power consumption
   - Dynamic power analysis
   - Power efficiency metrics

### Competitive Ranking

- Submissions are ranked by overall score
- Only submissions that beat the current challenge-wide best score receive rewards
- Weights are set to reward the highest-scoring submission
- Emission burning occurs when no submissions exceed quality thresholds
- The winner of a challenge will keep getting reward for specific time after challenge expiration

## Challenge Types

### Current Challenge Categories

1. **RISC-V based Processors**
2. **AI Accelerators**

## API Reference

### Miner Endpoints

```
GET  /api/v1/challenges/active              # Get active challenge
POST /api/v1/challenges/{id}/generate-submission-id  # Generate submission ID
POST /api/v1/challenges/{id}/submit         # Submit design solution
GET  /api/v1/challenges/{id}/submissions/hotkey/{hotkey}  # Check submissions
```

### Validator Endpoints

```
GET  /api/v1/challenges/{id}/batch/current  # Get current evaluation batch
GET  /api/v1/challenges/{id}/submissions/{submission_id}/download  # Download submission
POST /api/v1/challenges/{id}/submissions/{submission_id}/submit_score  # Submit evaluation
```

## Configuration

### Batch Management

The subnet uses a dynamic batch system:
- Submissions are grouped into evaluation batches
- Each batch has a download window (10 minutes) and evaluation window (20 minutes)
- Only one batch is exposed to validators at a time
- Batches transition: EXPOSED → EVALUATING → COMPLETED

## Security

### Authentication

- **Signature-based auth**: All API calls require Ed25519 signatures
- **Validator secrets**: Additional secret keys for validator endpoints
- **Hotkey verification**: Ensures submissions come from registered miners, and evaluated scores come from registered validators
- **Timestamp validation**: Prevents replay attacks

### Data Integrity

- **File hashing**: All submissions are verified with SHA256 hashes
- **Signature verification**: Using Bittensor's native signing methods

## Monitoring

### Health Checks

```bash
# Check challenge server health
curl http://challenge-server:8000/health
```

## Troubleshooting

### Common Issues

1. **Signature Verification Failed**
   - Ensure using Bittensor's native signing methods
   - Check timestamp accuracy (must be within 10 minutes)
   - Verify hotkey registration on subnet

2. **Submission Upload Failed**
   - Check ZIP file format and contents
   - Verify file size limits (typically 10MB)
   - Ensure proper authentication headers

3. **Validator Download Issues**
   - Confirm validator secret configuration
   - Check batch timing and availability
   - Verify network connectivity to challenge server

4. **EDA Tool Integration**
   - Ensure proper tool licensing and setup
   - Check environment variable configuration
   - Verify design constraints and timing requirements

## Contributing

### Submission Guidelines

1. Follow PEP 8 coding standards
2. Include comprehensive tests
3. Update documentation for new features
4. Ensure backward compatibility

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Discord**: [ChipForge - SN84](https://discord.com/channels/799672011265015819/1408463235082092564)

## Roadmap

### Upcoming Features

- Additional challenge categories
- Enhanced evaluation metrics
- Improved toolchain integration

### Version History

- **v1.0.0** - Initial release with challenge download, solution design, solution verification, submitting scores, and giving reward to winner.