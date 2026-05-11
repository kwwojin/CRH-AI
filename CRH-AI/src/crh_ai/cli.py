import argparse
import json
from .crh_core import CRHCoherenceBase


def main():
    parser = argparse.ArgumentParser(description='CRH Geometric Coherence Base CLI')
    parser.add_argument('--mode', choices=['process', 'reason', 'generate', 'state'], required=True,
                        help='Operation mode')
    parser.add_argument('--input', type=str, help='Input text for processing')
    parser.add_argument('--queries', type=str, help='JSON array of queries')
    parser.add_argument('--max-length', type=int, default=50, help='Max generation length')
    parser.add_argument('--embedding-dim', type=int, default=5, help='Embedding dimension')
    parser.add_argument('--coherence-threshold', type=float, default=0.8, help='Coherence threshold')
    
    args = parser.parse_args()
    
    crh = CRHCoherenceBase(
        embedding_dim=args.embedding_dim,
        coherence_threshold=args.coherence_threshold
    )
    
    if args.mode == 'process':
        if not args.input:
            print("Error: --input required for process mode")
            return
        
        result = crh.process_input(args.input)
        print(json.dumps(result, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
    
    elif args.mode == 'reason':
        if not args.input:
            print("Error: --input required for reason mode")
            return
        
        result = crh.reason(args.input)
        print(json.dumps(result, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
    
    elif args.mode == 'generate':
        if not args.input:
            print("Error: --input required for generate mode")
            return
        
        result = crh.generate(args.input, max_length=args.max_length)
        print(json.dumps(result, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
    
    elif args.mode == 'state':
        state = crh.get_system_state()
        print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
