from src.monitor import parse_args, region_from_args, run


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.config,
        region_from_args(arguments.region),
        arguments.manual_region,
        arguments.profile,
    )
