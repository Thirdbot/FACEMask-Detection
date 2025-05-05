import AppContainer from "../containers/AppContainer";
import PageContent from "../containers/PageContent";
import Sidebar from "../ui/Sidebar";
import Title from "../ui/Title";

const Settings = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="การตั้งค่า" />
        <p>
          Lorem ipsum dolor, sit amet consectetur adipisicing elit. A recusandae
          incidunt consectetur deleniti. Veritatis exercitationem placeat
          laborum itaque, unde, commodi quos asperiores repellat ad alias
          dolorem. Temporibus vero aliquid sequi?
        </p>
      </PageContent>
    </AppContainer>
  );
};

export default Settings;
