import matplotlib.pyplot as plt


def visualize_iterations(target, transformations, source, aligned):
    num_iterations = min(100, len(transformations))

    # Add more vertical space between rows
    fig, axes = plt.subplots(4, 6, figsize=(20, 25), gridspec_kw={"hspace": 0.3})

    fig.suptitle("ICP Algorithm Iterations", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < num_iterations:
            ax.scatter(
                target[:, 0], target[:, 1], c="b", label="Target", alpha=0.5, s=1
            )
            ax.scatter(
                transformations[i][:, 0],
                transformations[i][:, 1],
                c="r",
                label="Source",
                alpha=0.5,
                s=1,
            )

            ax.set_title(f"Iteration {i+1}")
            ax.legend(fontsize="x-small")
        else:
            ax.axis("off")

    for ax in axes.flat:
        ax.set_aspect("equal", "box")
        ax.tick_params(axis="both", which="major", labelsize=6)

    plt.tight_layout()
    plt.show()


def visualize_results(source, target, aligned):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.scatter(source[:, 0], source[:, 1], c="r", label="Source")
    ax1.scatter(target[:, 0], target[:, 1], c="b", label="Target")
    ax1.legend()
    ax1.set_title("Before Alignment")
    ax1.axis("equal")

    ax2.scatter(aligned[:, 0], aligned[:, 1], c="r", label="Aligned Source")
    ax2.scatter(target[:, 0], target[:, 1], c="b", label="Target")
    ax2.legend()
    ax2.set_title("After Alignment")
    ax2.axis("equal")

    ax3.scatter(source[:, 0], source[:, 1], c="r", alpha=0.5, label="Original Source")
    ax3.scatter(target[:, 0], target[:, 1], c="b", alpha=0.5, label="Target")
    ax3.scatter(aligned[:, 0], aligned[:, 1], c="g", alpha=0.5, label="Aligned Source")
    ax3.legend()
    ax3.set_title("Comparison")
    ax3.axis("equal")

    plt.tight_layout()
    plt.show()
