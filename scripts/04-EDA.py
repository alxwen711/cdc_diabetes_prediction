import click











@click.command()
@click.option(
    "--command",
    "-c", 
    type = str, 
    default = "count", 
    help = "Run a specific EDA command from the list of EDA tasks done in order: \n ['head', 'tail', 'describe', 'count', 'histogram', 'correlation']"
)
@click.option(
    "--save",
    "-s",
    type = str,
    default = None,
    help = "Specify a specific directory and filename to save a generated image to if the EDA command is among the following: \n ['count', 'histogram', 'correlation']"
)
def run_eda_function(command: str, save = None):
    match command:
        case "head":
            click.echo("Printing first few rows of training data...")
        case "tail":
            click.echo("Printing last few rows of training data...")
        case "describe":
            click.echo("Printing description of training data...")
        case "count":
            click.echo("Creating basic bar plot for count of target labels...")
        case "histogram":
            click.echo("Creating histograms of all features...") # Might be adjusted?
        case "correlation":
            click.echo("Creating feature-feature correlation plot on training data...")
        case _:
            click.echo("Command is not recognized from the options ['head', 'tail', 'describe', 'count', 'histogram', 'correlation'], no action performed")
            return
    click.echo(f"Command {command} compeleted as expected.")
    if save != None and command in ["count", "histogram", "correlation"]:
        click.echo(f"Saving plot to {save}...")
        click.echo(f"Plot has been saved to {save}.")

if __name__ == "__main__":
    run_eda_function()



