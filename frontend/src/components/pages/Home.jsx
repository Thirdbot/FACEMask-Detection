import React from "react";
import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";
import Title from "../ui/Title";

const Home = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="หน้าหลัก" />
      </PageContent>
    </AppContainer>
  );
};

export default Home;
