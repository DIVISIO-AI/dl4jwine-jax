package divisio.dl4jwine;

import java.io.File;
import java.io.IOException;

public interface Trainer {

    /**
     * @return a tag to be used in creating the name of the folder with the training result. This is necessary so we
     * can later compare different training results.
     */
    String getTag();

    /**
     * train for one epoch
     */
    void train() throws Exception;

    /**
     * run validation
     */
    void validate() throws Exception;

    /**
     * run testing
     */
    void test() throws Exception;

    /**
     * load state from file, used to resume training
     * @param file not null, file with training data
     */
    void loadState(final File file) throws IOException;

    /**
     * write state to given file
     * @param file
     */
    void saveState(final File file) throws IOException;
}
