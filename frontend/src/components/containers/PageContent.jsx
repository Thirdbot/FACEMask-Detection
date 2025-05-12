const PageContent = ({ children, className }) => {
  return (
    <div
      className={`px-10 pt-14 ${className} flex flex-col items-center justify-start w-full`}
    >
      {children}
    </div>
  );
};

export default PageContent;
